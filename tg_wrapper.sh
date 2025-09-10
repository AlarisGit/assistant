while true
do
  echo "======================================================"
  date
  echo "------------------------------------------------------"
  git pull
  pip install -r requirements.txt
  python src/tg.py
  sleep 10
done

