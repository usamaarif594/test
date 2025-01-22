import streamlit as st
import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime
import plotly.express as px
import numpy as np
import riskfolio as rp
import skfolio as sk
from skfolio import RiskMeasure
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.preprocessing import prices_to_returns

st.title("Racial Harm Portfolio Analyzer")

# Initialize session state for storing the DataFrame
if 'portfolio_df' not in st.session_state:
    st.session_state.portfolio_df = pd.DataFrame(columns=[
        'Stock', 'Units', 'Purchase Date', 'Purchase Price', 'Current Price',
        'Initial Investment', 'Current Value', 'Gain/Loss', 'Gain/Loss %', 
        'Portfolio Allocation', 'GICS Sector', 'Normalized Harm Score'
    ])

# Function to update portfolio allocation
def update_portfolio_allocation(df):
    if not df.empty:
        total_value = df['Current Value'].str.replace('$', '').astype(float).sum()
        df.loc[:, 'Portfolio Allocation'] = df['Current Value'].str.replace('$', '').astype(float) / total_value * 100
        df.loc[:, 'Portfolio Allocation'] = df['Portfolio Allocation'].apply(lambda x: f"{x:.2f}%")
    return df

# Function to calculate Mean Harm Score 
def calculate_mean_harm_score(portfolio_df):
    if 'Normalized Harm Score' in portfolio_df.columns:
        return portfolio_df['Normalized Harm Score'].astype(float).mean()
    else:
        raise ValueError("The DataFrame does not contain a 'Normalized Harm Score' column.")

# Function to get GICS Sector
def get_gics_sector(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info.get('sector', 'N/A')
    except:
        return 'N/A'

# Function to get normalized scores from SQLite database based on sector
def get_normalized_score(sector):
    conn = sqlite3.connect('nycprocurement.db')
    query = "SELECT Normalized_Score_2 FROM stockracialharm WHERE sector = ?"
    scores = pd.read_sql_query(query, conn, params=(sector,))
    conn.close()
    
    return scores.iloc[0]['Normalized_Score_2'] if not scores.empty else None

# Function to get normalized scores from SQLite database based on sector
def get_normalized_score2(sector):
    conn = sqlite3.connect('nycprocurement.db')
    query = "SELECT Normalized_Score_2 FROM stockracialharm WHERE sector = ?"
    scores = pd.read_sql_query(query, conn, params=(sector,))
    conn.close()
    return scores.iloc[0]['Normalized_Score_2'] if not scores.empty else None

    
#     return scores.iloc[0]['normal_score_graph1'] if not scores.empty else None

# Function to optimize portfolio allocation
def optimize_portfolio_allocation(df, max_harm_score):
    # Prepare returns data with robust error handling
    returns_data = (
        df['Gain/Loss %']
        .astype(str)
        .str.replace('%', '')
        .replace('', '0')
        .astype(float)
        .to_numpy()
        .reshape(-1, 1) 
        / 100
    )
    
    harm_scores = df['Normalized Harm Score'].astype(float)

    # Create a portfolio object and set constraints
    weights = np.ones(len(df)) / len(df)  # Initial equal weights
    port = rp.Portfolio(returns=returns.values)

    # Optimize using Mean-Variance Optimization (MVO)
    optimized_weights = port.optimization(model='Classic', objective='max_sharpe')

    # Apply harm score constraint manually
    while True:
        weighted_harm_score = np.dot(harm_scores, optimized_weights)
        if weighted_harm_score <= max_harm_score:
            break
        # If constraint is violated, reduce weights proportionally
        optimized_weights *= (max_harm_score / weighted_harm_score)

    # Update portfolio allocation based on optimized weights
    df.loc[:, 'Portfolio Allocation'] = optimized_weights * 100  # Convert to percentage
    return df

# Sidebar for adding stocks
with st.sidebar:
    st.header("Add Stock to Portfolio")
    with st.form("stock_form"):
        ticker = st.text_input("Enter stock ticker (e.g., AAPL, GOOGL)")
        units = st.number_input("Enter number of units", min_value=1, step=1)
        transaction_date = st.date_input("Select transaction date")
        submit_button = st.form_submit_button(label="Add to Portfolio")

# Main application logic
if submit_button:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=transaction_date)
        
        if hist.empty:
            st.error("No data available for the selected date. Please choose a valid trading day.")
        else:
            purchase_price = hist.iloc[0]['Close']
            current_price = stock.info['currentPrice']
            
            initial_investment = purchase_price * units
            current_value = current_price * units
            gain_loss = current_value - initial_investment
            gain_loss_percentage = (gain_loss / initial_investment) * 100
            
            gics_sector = get_gics_sector(ticker)
            normalized_score = get_normalized_score(gics_sector)
            # normalized_score2 = get_normalized_score2(gics_sector)
            
            new_row = pd.DataFrame({
                'Stock': [ticker],
                'Units': [units],
                'Purchase Date': [transaction_date],
                'Purchase Price': [f"${purchase_price:.2f}"],
                'Current Price': [f"${current_price:.2f}"],
                'Initial Investment': [f"${initial_investment:.2f}"],
                'Current Value': [f"${current_value:.2f}"],
                'Gain/Loss': [f"${gain_loss:.2f}"],
                'Gain/Loss %': [gain_loss_percentage],
                'Portfolio Allocation': ["0.00%"],
                'GICS Sector': [gics_sector],
                'Normalized Harm Score': [normalized_score],
                # 'Normalized Harm Score2': [normalized_score2]
            })
            
            st.session_state.portfolio_df = pd.concat([st.session_state.portfolio_df, new_row], ignore_index=True)
            st.session_state.portfolio_df = update_portfolio_allocation(st.session_state.portfolio_df)
            
            st.success(f"Added {ticker} to your portfolio with sector '{gics_sector}' and normalized score '{normalized_score}'.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Optimize portfolio if stocks are present and button is clicked
if not st.session_state.portfolio_df.empty and st.button("Optimize Portfolio", key="optimize_portfolio_button"):  # Unique key added
    min_harm_score = st.sidebar.number_input("Minimum Average Harm Score", min_value=0.0, step=0.01)
    
    # Calculate mean harm score from the portfolio DataFrame
    mean_harm_score = calculate_mean_harm_score(st.session_state.portfolio_df)

    # Check if mean_harm_score was calculated successfully
    if mean_harm_score is None:
        st.warning("The DataFrame does not contain a 'Normalized Harm Score' column.")
    else:
        if mean_harm_score >= min_harm_score:
            # Prepare returns data for optimization from "Gain/Loss %" column
            returns_data = st.session_state.portfolio_df['Gain/Loss %'].astype(str).str.replace('%', '').astype(float).to_numpy().reshape(-1, 1) / 100

            
            # Check shape of returns_data
            print("Returns data shape:", returns_data.shape)

            # Reshape if necessary
            if returns_data.ndim == 1:
                returns_data = returns_data.values.reshape(-1, 1)  # Reshape to be 2D
            
            # Initialize MeanRisk model for optimization
            model = MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_RATIO, risk_measure=RiskMeasure.VARIANCE)

            # Fit the model with the returns data (reshaping if necessary)
            model.fit(returns_data)

            # Get optimized weights
            optimized_weights = model.weights_

            ## Update session state with optimized allocations
            st.session_state.portfolio_df['Portfolio Allocation'] = [optimized_weights[i] for i in range(len(st.session_state.portfolio_df))]

            st.success("Portfolio optimized based on the minimum average harm score.")
            
            # Display the new DataFrame with optimized allocations above the plots
            st.subheader("Optimized Portfolio Allocations")
            st.dataframe(st.session_state.portfolio_df[['Stock', 'Portfolio Allocation']])
            
        else:
            st.warning(f"Mean harm score {mean_harm_score:.2f} is below the minimum allowed score of {min_harm_score:.2f}.")

# Display the portfolio table and other visualizations...
if not st.session_state.portfolio_df.empty:
    st.subheader("Public Equity Portfolio")
    st.dataframe(st.session_state.portfolio_df)


    # Create two columns for side-by-side layout for visualizations
    col1, col2 = st.columns(2)

    # Create a doughnut chart for normalized harm scores in the first column
    # Assuming 'Units' is the column name for the number of units
    units_df = st.session_state.portfolio_df[['Stock', 'Units']]

    # Check if 'Units' column is not null
    if not units_df['Units'].isnull().all():
        units_df['Units'] = units_df['Units'].astype(float)

        # Calculate total units
        total_units = units_df['Units'].sum()

        # Create a new column for percentage
        units_df['Percentage'] = (units_df['Units'] / total_units) * 100

        # Create doughnut chart using the percentage
        fig1 = px.pie(units_df,
                    names='Stock',
                    values='Percentage',
                    hole=0.4,
                    title="Stock Portfolio Units as Percentage of Total",
                    labels={'Percentage': 'Percentage of Total Units'})

        with col1:
            st.plotly_chart(fig1)

    # Create a doughnut chart that calculate normalized harm score * number of units and percentage of total harm scores times units.
    
    total_harm_score_units = (st.session_state.portfolio_df['Normalized Harm Score2'].astype(float) *
                               st.session_state.portfolio_df['Units']).sum()
    
    st.session_state.portfolio_df['Harm Score Contribution (%)'] = (
       (st.session_state.portfolio_df['Normalized Harm Score2'].astype(float) *
         st.session_state.portfolio_df['Units']) / total_harm_score_units * 100).fillna(0)

    contribution_data = st.session_state.portfolio_df[['Stock', 'Harm Score Contribution (%)']]
    
    if not contribution_data['Harm Score Contribution (%)'].isnull().all():
       contribution_data['Harm Score Contribution (%)'] = contribution_data['Harm Score Contribution (%)'].astype(float)
    
       fig2 = px.pie(contribution_data, 
                      names='Stock', 
                      values='Harm Score Contribution (%)',
                      title="Portfolio Harm Contribution by Stock",
                      labels={'Harm Score Contribution (%)': 'Contribution (%)'},
                      hole=0.4)  

       with col2:
           st.plotly_chart(fig2)

# Option to remove stocks from portfolio in sidebar
with st.sidebar:
   st.header("Remove Stocks from Portfolio")
   stocks_to_remove = st.multiselect("Select stocks to remove", 
                                      options=st.session_state.portfolio_df['Stock'].unique())

   if st.button("Remove Selected Stocks", key="remove_stocks_button"):  # Unique key added
       st.session_state.portfolio_df = st.session_state.portfolio_df[
           ~st.session_state.portfolio_df['Stock'].isin(stocks_to_remove)]
        
       # Update portfolio allocation after removal
       st.session_state.portfolio_df = update_portfolio_allocation(st.session_state.portfolio_df)
       st.success("Selected stocks removed from portfolio.")

# Optimization Button in Sidebar
with st.sidebar:
    st.header("Optimize Portfolio")
    if not st.session_state.portfolio_df.empty and st.button("Optimize Portfolio", key="sidebar_optimize_portfolio_button"):  # Unique key added
        max_harm_score = st.sidebar.number_input("Maximum Average Harm Score", min_value=0.0, step=0.01)
        
        mean_harm_score = st.session_state.portfolio_df['Normalized Harm Score'].astype(float).mean()
        
        if mean_harm_score <= max_harm_score:
            st.session_state.portfolio_df = optimize_portfolio_allocation(st.session_state.portfolio_df, max_harm_score)
            st.success("Portfolio optimized based on the maximum average harm score.")
            
        else:
            st.warning(f"Mean harm score {mean_harm_score:.2f} exceeds the maximum allowed score of {max_harm_score:.2f}.")
    

# Portfolio Summary and Analysis Section
if not st.session_state.portfolio_df.empty:
   portfolio_value = st.session_state.portfolio_df['Current Value'].str.replace('$', '').astype(float).sum()
    
   # Display total portfolio value
   st.metric("Total Portfolio Value", f"${portfolio_value:.2f}")
else:
   st.info("Add stocks to your portfolio to see analysis.")


