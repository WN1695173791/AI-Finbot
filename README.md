# AI-Finbot
An intelligent portfolio management bot

## How to use it

1. **Clone repository:** Install git and clone this repo in your project directory.

```bash
git clone https://github.com/AkibMashrur/AI-Finbot
```

2. **Set API keys:** Create a .env file and insert binance keys (follow **.env.example** file for template). 

3. **Install dependencies:** Create a new virtual environment in your directory and install dependencies into your virtual environment in the following code. Additional commands you may need for installing pip/venv/notebook and creating virtual environments is given in **scripts/installation** folder. 

On linux shell:
```bash
source venv/bin/activate
pip install -r requirements.txt
```
or on windows terminal:
```terminal
.\venv\Scripts\activate
pip install -r requirements.txt
```

4. **Start bot:** Run main.py file to start the bot

```bash
python main.py
```
or on windows:
```bash
py main.py
```

5. **Train or backtest the bot:** Find a detailed guide on training/backtesting the bot in **notebook.ipynb** file.