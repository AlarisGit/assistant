def _strip_bom(s: str | None) -> str | None:
    """Remove leading UTF-8 BOM (\ufeff) if present."""
    if isinstance(s, str) and s.startswith("\ufeff"):
        return s.lstrip("\ufeff")
    return s


def _remove_bom_all(s: str | None) -> str | None:
    """Remove all occurrences of U+FEFF from the string."""
    if isinstance(s, str):
        return s.replace("\ufeff", "")
    return s


def _html_to_markdown_with_headings(html: str, page_title: str | None = None) -> str:
    """Convert HTML to Markdown, promoting likely section title elements to headings.
    - Elements with role="heading" (aria-level honored) become h1..h6
    - Elements with common title classes become h2
    - Uses ATX heading style (###)
    """
    try:
        html = _remove_bom_all(_strip_bom(html)) or ""
        soup = BeautifulSoup(html, "html.parser")

        # Remove script/style/noscript elements entirely to avoid CSS/JS leftovers
        for t in ("script", "style", "noscript"):
            for tag in soup.find_all(t):
                tag.decompose()

        # Promote explicit ARIA headings
        for el in soup.find_all(attrs={"role": "heading"}):
            try:
                level = int(el.get("aria-level", 2))
            except ValueError:
                level = 2
            level = max(1, min(level, 6))
            el.name = f"h{level}"

        # Promote elements with heading-ish classes, ids, or data attributes
        class_pattern = re.compile(
            r"(?:^|\b)(title|section-title|chapter-title|doc-title|page-title|header-title|toc-title|section_name|sectionname|sectionheader|section-header|block-title)(?:\b|$)",
            re.IGNORECASE,
        )
        id_pattern = re.compile(r"^(section|chapter|part|toc)[-_]?(title|name|header)?$", re.IGNORECASE)
        data_attrs = ["data-title", "data-section-title", "data-header"]
        for el in soup.find_all(True):
            # Already a heading
            if el.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                continue
            has_heading_hint = False
            # class hint
            cls = " ".join(el.get("class", [])).strip()
            if cls and class_pattern.search(cls):
                has_heading_hint = True
            # id hint
            if not has_heading_hint:
                eid = el.get("id")
                if eid and id_pattern.search(eid):
                    has_heading_hint = True
            # data-* hint
            if not has_heading_hint:
                for da in data_attrs:
                    if el.get(da):
                        has_heading_hint = True
                        break
            if has_heading_hint:
                el.name = "h2"

        # Convert to Markdown with ATX heading style and strip non-content areas
        md = _html_to_md(
            str(soup),
            heading_style="ATX",
            strip=["script", "style", "nav", "header", "footer", "noscript"],
        ).strip()
        if page_title:
            page_title = (_remove_bom_all(_strip_bom(page_title)) or page_title).strip()
            # If there is already an H1 equal to page_title, skip prepending
            existing_h1 = soup.find(["h1"])  # after promotions
            existing_h1_text = existing_h1.get_text(strip=True) if existing_h1 else None
            if not (existing_h1_text and existing_h1_text.strip() == page_title):
                md = f"# {page_title}\n\n" + md
        # Post-process: if first non-empty line is '# Title' and next non-empty equals 'Title', drop duplicate
        lines = md.splitlines()
        def first_nonempty_idx(lst):
            for i, ln in enumerate(lst):
                if ln.strip():
                    return i
            return -1
        i1 = first_nonempty_idx(lines)
        if i1 != -1 and lines[i1].startswith("# "):
            title_text = lines[i1][2:].strip()
            # find next non-empty line after i1
            i2 = -1
            for j in range(i1 + 1, len(lines)):
                if lines[j].strip():
                    i2 = j
                    break
            if i2 != -1 and lines[i2].strip() == title_text:
                lines.pop(i2)
        md = "\n".join(lines)
        # TOC formatting improvements:
        # 1) Remove javascript:void(...) targets left by nav elements
        md = re.sub(r"\(javascript:[^)]*\)", "", md)
        # 2) Split multiple inline bullets onto separate lines for readability
        for marker in (" - [", " * [", " + ["):
            md = md.replace(marker, "\n- [")
        return _remove_bom_all(md) or md
    except Exception as e:
        logger.debug(f"Markdown conversion fallback: {e}")
        return _remove_bom_all(_html_to_md(_remove_bom_all(_strip_bom(html)) or "", heading_style="ATX", strip=["script", "style"]).strip()) or ""

import config
import logging
import atexit
import os
from urllib.parse import urljoin, urlparse, urldefrag
import json
import time
from collections import deque
import re
import requests
from bs4 import BeautifulSoup
import urllib.robotparser as robotparser
from markdownify import markdownify as _html_to_md
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_urls = {}
# Root directories for storing crawled content
html_root_dir = os.path.join(config.DATA_DIR, "html")
images_root_dir = os.path.join(config.DATA_DIR, "images")
md_root_dir = os.path.join(config.DATA_DIR, "md")
os.makedirs(html_root_dir, exist_ok=True)
os.makedirs(images_root_dir, exist_ok=True)
os.makedirs(md_root_dir, exist_ok=True)
# A single combined index for all crawls
ulrs_file = os.path.join(html_root_dir, "urls.json")

DEFAULT_CRAWL_DELAY = 0.5  # seconds between requests
DEFAULT_MAX_PAGES = 10000   # safety cap
REQUEST_TIMEOUT = 15       # seconds
USER_AGENT = "ProductManualCrawler/1.0 (+https://example.com)"


def _init_robot_parser(base_url: str) -> robotparser.RobotFileParser:
    parsed = urlparse(base_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
    except Exception as e:
        logger.warning(f"Failed to read robots.txt at {robots_url}: {e}")
    return rp


def _same_domain(url: str, base_netloc: str) -> bool:
    try:
        return urlparse(url).netloc == base_netloc
    except Exception:
        return False


def _normalize_link(base_url: str, link: str) -> str | None:
    if not link:
        return None
    link = link.strip()
    # Skip non-http(s) schemes and javascript/mailto/tel
    if link.startswith(("javascript:", "mailto:", "tel:", "#")):
        return None
    abs_url = urljoin(base_url, link)
    # Remove fragment
    abs_url, _ = urldefrag(abs_url)
    return abs_url


def _is_html_response(resp: requests.Response) -> bool:
    ctype = resp.headers.get("Content-Type", "").lower()
    return "text/html" in ctype or "application/xhtml+xml" in ctype


def _save_page(target_dir: str, url: str, html: str) -> str:
    """Save HTML content under the given target_dir (already per-domain), mirroring site path. Returns file path."""
    parsed = urlparse(url)
    # target_dir is already the per-domain directory
    domain_dir = target_dir
    path = parsed.path
    if not path or path.endswith("/"):
        path = path + "index.html"
    filename = path.lstrip("/")
    # Encode query into filename if present
    if parsed.query:
        safe_query = re.sub(r"[^A-Za-z0-9._-]", "_", parsed.query)
        if filename.endswith(".html"):
            filename = filename[:-5] + f"_{safe_query}.html"
        else:
            filename = filename + f"_{safe_query}.html"
    full_path = os.path.join(domain_dir, filename)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(html)
    return full_path


def _save_markdown(target_dir: str, url: str, markdown_text: str) -> str:
    """Save Markdown content under the given target_dir (already per-domain), mirroring site path. Returns file path."""
    parsed = urlparse(url)
    domain_dir = target_dir
    path = parsed.path
    if not path or path.endswith("/"):
        path = path + "index.md"
    # Replace extension with .md
    base, _ext = os.path.splitext(path)
    filename = base.lstrip("/") + ".md"
    # Encode query into filename if present
    if parsed.query:
        safe_query = re.sub(r"[^A-Za-z0-9._-]", "_", parsed.query)
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{safe_query}{ext or ''}"
    full_path = os.path.join(domain_dir, filename)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    clean_md = _remove_bom_all(_strip_bom(markdown_text)) or ""
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(clean_md)
    return full_path


_IMAGE_EXT_REGEX = re.compile(r"\.(png|jpe?g|gif|bmp|webp|svg|tiff?|ico|heic|heif)$", re.IGNORECASE)


def _is_image_request(url: str, headers: dict) -> bool:
    ctype = headers.get("Content-Type", "").lower() if headers else ""
    if ctype.startswith("image/"):
        return True
    # fallback by extension
    parsed = urlparse(url)
    return bool(_IMAGE_EXT_REGEX.search(parsed.path))


def _save_image(target_dir: str, url: str, content: bytes, headers: dict) -> str:
    parsed = urlparse(url)
    # target_dir is already the per-domain directory
    domain_dir = target_dir
    path = parsed.path
    if not path or path.endswith("/"):
        # derive extension from content-type
        ext = ""
        ctype = (headers or {}).get("Content-Type", "").lower()
        if ctype.startswith("image/"):
            ext = "." + ctype.split("/", 1)[1].split(";")[0].strip()
            # normalize jpeg/jpg
            if ext == ".jpeg":
                ext = ".jpg"
        path = (path or "/") + f"image{ext or '.bin'}"
    filename = path.lstrip("/")
    # Encode query into filename if present
    if parsed.query:
        safe_query = re.sub(r"[^A-Za-z0-9._-]", "_", parsed.query)
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{safe_query}{ext or ''}"
    full_path = os.path.join(domain_dir, filename)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "wb") as f:
        f.write(content)
    return full_path


def _fetch(url: str) -> tuple[int | None, str | None, dict]:
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        hdrs = dict(resp.headers)
        if not _is_html_response(resp):
            return resp.status_code, None, hdrs
        # Decode bytes robustly, honoring BOM when present
        raw = resp.content or b""
        text = None
        try:
            # If starts with UTF-8 BOM, decode with utf-8-sig to strip it
            if raw.startswith(b"\xef\xbb\xbf"):
                text = raw.decode("utf-8-sig", errors="replace")
            else:
                enc = resp.encoding or requests.utils.get_encoding_from_headers(resp.headers) or "utf-8"
                # Normalize common enc labels
                if enc.lower() in ("utf8", "utf-8"):
                    # ensure BOM would be removed if present (handled above)
                    text = raw.decode("utf-8", errors="replace")
                else:
                    try:
                        text = raw.decode(enc, errors="replace")
                    except LookupError:
                        text = raw.decode("utf-8", errors="replace")
        except Exception:
            text = raw.decode("utf-8", errors="replace")
        # Final cleanup of U+FEFF anywhere
        text = _remove_bom_all(_strip_bom(text)) or ""
        return resp.status_code, text, hdrs
    except requests.RequestException as e:
        logger.warning(f"Request failed for {url}: {e}")
        return None, None, {}


def _extract_links(base_url: str, html: str) -> list[str]:
    links = []
    try:
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            norm = _normalize_link(base_url, a.get("href"))
            if norm:
                links.append(norm)
        # Also include image sources as links for extraction convenience
        for img in soup.find_all("img", src=True):
            norm = _normalize_link(base_url, img.get("src"))
            if norm:
                links.append(norm)
    except Exception as e:
        logger.warning(f"Failed to parse HTML from {base_url}: {e}")
    return links


def _extract_title_description(html: str) -> tuple[str | None, str | None]:
    """Extract <title> and description from meta tags.
    Checks standard description, og:description, and itemprop variants.
    """
    try:
        soup = BeautifulSoup(_remove_bom_all(_strip_bom(html)) or "", "html.parser")
        # Title
        title = None
        if soup.title and soup.title.string:
            title = _remove_bom_all(_strip_bom(soup.title.string)).strip()
        # Description candidates
        description = None
        # name="description"
        tag = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
        if tag and tag.get("content"):
            description = tag.get("content").strip()
        # property="og:description"
        if not description:
            tag = soup.find("meta", attrs={"property": re.compile(r"^og:description$", re.I)})
            if tag and tag.get("content"):
                description = tag.get("content").strip()
        # itemprop="description"
        if not description:
            tag = soup.find("meta", attrs={"itemprop": re.compile(r"^description$", re.I)})
            if tag and tag.get("content"):
                description = tag.get("content").strip()
        return title or None, description or None
    except Exception as e:
        logger.debug(f"Failed to extract title/description: {e}")
        return None, None


def crawl_site(base_url: str, start_path: str = "/", max_pages: int = DEFAULT_MAX_PAGES, delay: float = DEFAULT_CRAWL_DELAY):
    """
    Crawl within the same domain as base_url, starting from start_path.
    Saves HTML under DATA_DIR/html/<domain> and images under DATA_DIR/images/<domain>.
    Records metadata in urls.json under DATA_DIR/html/.
    """
    global _urls
    base_url = base_url.rstrip("/")
    if not base_url.startswith("http://") and not base_url.startswith("https://"):
        raise ValueError("DOC_BASE_URL must start with http:// or https://")

    parsed_base = urlparse(base_url)
    base_netloc = parsed_base.netloc
    start_url = urljoin(base_url + "/", start_path.lstrip("/"))

    rp = _init_robot_parser(base_url)

    q = deque([start_url])
    seen = set(_urls.keys())  # continue from previous state

    # Prepare per-site storage directories
    site_html_dir = os.path.join(html_root_dir, base_netloc)
    site_images_dir = os.path.join(images_root_dir, base_netloc)
    site_md_dir = os.path.join(md_root_dir, base_netloc)
    os.makedirs(site_html_dir, exist_ok=True)
    os.makedirs(site_images_dir, exist_ok=True)
    os.makedirs(site_md_dir, exist_ok=True)

    pages_crawled = 0
    while q and pages_crawled < max_pages:
        url = q.popleft()
        if url in seen:
            continue

        if not _same_domain(url, base_netloc):
            continue

        if not rp.can_fetch(USER_AGENT, url):
            logger.info(f"Blocked by robots.txt: {url}")
            seen.add(url)
            _urls[url] = {
                "ts": time.time(),
                "status": "blocked_by_robots",
            }
            continue

        status_code, html, headers = _fetch(url)
        meta = {
            "ts": time.time(),
            "status_code": status_code,
            "content_type": headers.get("Content-Type"),
            "saved_path": None,
            "out_links": [],
            "images_saved": [],
            "title": None,
            "description": None,
        }

        if html:
            save_path = _save_page(site_html_dir, url, html)
            meta["saved_path"] = os.path.relpath(save_path, html_root_dir)
            # Extract page title and description (before MD conversion)
            t, d = _extract_title_description(html)
            meta["title"] = t
            meta["description"] = d
            # Convert HTML to Markdown (promote section titles to headings) and save
            md_text = _html_to_markdown_with_headings(html, page_title=t)
            md_path = _save_markdown(site_md_dir, url, md_text)
            meta["saved_md_path"] = os.path.relpath(md_path, md_root_dir)
            links = _extract_links(url, html)
            # Filter to same domain and normalize
            next_links = []
            image_links = []
            for l in links:
                if _same_domain(l, base_netloc):
                    # Separate image links
                    if _is_image_request(l, {}):
                        image_links.append(l)
                    else:
                        next_links.append(l)
            meta["out_links"] = next_links
            # Download images (same domain only), respecting robots.txt
            for img_url in image_links:
                if not rp.can_fetch(USER_AGENT, img_url):
                    logger.debug(f"Blocked image by robots.txt: {img_url}")
                    continue
                try:
                    r = requests.get(img_url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT, stream=True)
                    content = r.content
                    headers_final = dict(r.headers)
                    if _is_image_request(img_url, headers_final):
                        img_path = _save_image(site_images_dir, img_url, content, headers_final)
                        meta["images_saved"].append(os.path.relpath(img_path, images_root_dir))
                except requests.RequestException as e:
                    logger.debug(f"Failed to download image {img_url}: {e}")
            for l in next_links:
                if l not in seen:
                    q.append(l)
        else:
            logger.debug(f"Skipping non-HTML or failed fetch: {url}")

        _urls[url] = meta
        seen.add(url)
        pages_crawled += 1
        if q and delay > 0:
            time.sleep(delay)

def load_urls(path):
    global _urls
    if os.path.exists(path):
        with open(path, "r") as f:
            _urls = json.load(f)

def save_urls(path):
    global _urls
    with open(path, "w") as f:
        json.dump(_urls, f, indent=2, sort_keys=True)

atexit.register(save_urls, ulrs_file)

load_urls(ulrs_file)

if __name__ == '__main__':
    logger.info(f"Starting crawl for {config.DOC_BASE_URL}")
    crawl_site(config.DOC_BASE_URL, "/")