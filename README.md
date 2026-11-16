NLP Game dev assistant
 open powershell
create and active a virtual; environmemnt

python -m venv .venv
.\.venv\Scripts\Activate.ps1

Upgrade and pip install the required packages

python -m pip install --upgrade pip
pip install -r requirements.txt

Usage

1.Run the scraping script
python scrap.py

2. Load the data 
python load.py

3.Launch the streamlit dashboard
streamlit run app.py
