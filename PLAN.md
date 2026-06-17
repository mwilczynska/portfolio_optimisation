PLAN

This project's aim is to build an investment portfolio optimisation algorithm in python.

Data
I have previously constructed daily timeseries datasets of various assets that we will use for the portfolio optimisation. 

The datasets are located in:
C:\Datascience\projects\finance\financial_datasets\data\processed

You can read about the methodologies used to create the datasets in:
C:\Datascience\projects\finance\financial_datasets\docs

Previous work:
I have created a portfolio backtesting script in:
C:\Datascience\python\investment_tracker

The current file is:
C:\Datascience\python\investment_tracker\portfolio_plot15.py

Stick to the syntax and format and design within that script.

Requirements

Create an optimisation algorithm for various portfolio weights of these assets:
USLCAP3x
LTT3x
ITT3x
GOLDPM2x
COMM

That is, I would like to test all possible portfolios, at 5% increments, of the above assets.

For each portfolio we will backtest the entire timeseries (from 1970).

With each run we want to log many financial statistics such as CAGR, Std Dev, TWRR, Sharpe ratio, Max drawdown, Ulcer Index - and please add many more.

I would also like options within this script to set a minimum floor for any of the assets and maximum roof for any of the assets. Make this user editable within the script.

I would also like a max drawdown user limit to be able to be set

Create a table of results, ranked by CAGR (but this should also be user editable)

Additionally I would like to be able to plot CAGR vs. max drawdown with each portfolio as a dot. I'm not sure yet what the best method is for being able to ientify the portfolio of any one portfolio within the plot - perhaps an interactable plot? You decide.

Housekeeping:
Create an AGENTS.md file and a CLAUDE.md file. The files should be hard linked such that any changes to one file will be reflected in the other automatically.

Create a LOG.md file of any work that has been done.

Create a HANDOVER.md file. This file will have relevant notes for any agent that continues to work on this project.

Check if any of these files need updating after each piece of work is finished/committed/pushed to repo. Commit and push after each chunk of work is complete.

Repo is here:
https://github.com/mwilczynska/portfolio_optimisation.git