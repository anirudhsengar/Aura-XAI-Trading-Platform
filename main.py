import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import sys
import os
from pathlib import Path
import numpy as np

warnings.filterwarnings('ignore')

# Add backend directory to path
backend_path = Path(__file__).parent / "backend"
sys.path.append(str(backend_path))

# Import backend modules
try:
    from data_manager import DataManager
    from feature_engine import FeatureEngine
    from strategies import StrategyFactory
    from backtester import BacktestEngine
    from explainer import Explainer
    from utils import DataValidator, LoggingUtils
except ImportError as e:
    st.error(f"Error importing backend modules: {e}")
    st.stop()

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'current_strategy' not in st.session_state:
    st.session_state.current_strategy = None

@st.cache_data
def load_data(symbol, start_date, end_date):
    """Load and cache market data."""
    try:
        dm = DataManager()
        market_data = dm.fetch_market_data(symbol, start_date, end_date)
        news_data = dm.fetch_news_data(symbol, start_date, end_date)
        return market_data, news_data, None
    except Exception as e:
        return None, None, str(e)

@st.cache_data
def process_features(market_data, news_data, symbol):
    """Process features and cache results."""
    try:
        fe = FeatureEngine()
        
        # Calculate technical indicators
        technical_data = fe.calculate_technical_indicators(market_data, symbol)
        
        # Calculate sentiment features
        sentiment_data = fe.calculate_sentiment_features(news_data, symbol) if not news_data.empty else pd.DataFrame()
        
        # Combine features
        combined_data = fe.combine_features(technical_data, sentiment_data)
        
        return combined_data, None
    except Exception as e:
        return None, str(e)

def run_backtest(strategy_name, strategy_params, data, symbol):
    """Run backtest with given strategy."""
    try:
        # Create strategy
        strategy = StrategyFactory.create_strategy(strategy_name, strategy_params)

        # Initialize backtester
        backtester = BacktestEngine(
            initial_capital=100000,
            commission_rate=0.001,
            slippage_rate=0.0005
        )
        
        # Run backtest
        results = backtester.run_backtest(strategy, data, symbol)
        
        return results, strategy, None
    except Exception as e:
        import traceback
        error_msg = f"Error in backtest: {str(e)}\n{traceback.format_exc()}"
        return None, None, error_msg

def main():
    st.set_page_config(
        page_title="Aura - Explainable AI Trading Platform",
        page_icon="🔮",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Aura - XAI Trading Platform")
    st.markdown("*Demystifying algorithmic trading through transparent AI explanations*")
    
    # Sidebar for user inputs
    st.sidebar.header("Configuration")
    
    # Asset Selection
    st.sidebar.subheader("Asset Selection")
    symbol = st.sidebar.text_input("Stock Symbol", value="AAPL", help="Enter stock ticker (e.g., AAPL, GOOGL)")
    
    # Validate symbol
    if symbol and not DataValidator.validate_stock_symbol(symbol):
        st.sidebar.error("Invalid stock symbol format")
        return
    
    # Date Range Selection
    st.sidebar.subheader("Date Range")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())
    
    # Validate date range
    if not DataValidator.validate_date_range(datetime.combine(start_date, datetime.min.time()), 
                                           datetime.combine(end_date, datetime.min.time())):
        st.sidebar.error("Invalid date range")
        return
    
    # Strategy Selection
    st.sidebar.subheader("Strategy Selection")
    strategy_options = {
        "Simple MA Crossover": "simple_ma",
        "Mean Reversion": "mean_reversion",
        "Momentum": "momentum", 
        "Multi-Factor": "multi_factor"
    }
    selected_strategy_name = st.sidebar.selectbox("Choose Strategy", list(strategy_options.keys()))
    selected_strategy = strategy_options[selected_strategy_name]
    
    # Strategy Parameters
    st.sidebar.subheader("Strategy Parameters")
    strategy_params = {}
    
    if selected_strategy == "simple_ma":
        strategy_params = {
            'fast_ma': st.sidebar.slider("Fast MA Period", 5, 20, 10),
            'slow_ma': st.sidebar.slider("Slow MA Period", 15, 50, 20)
        }
    elif selected_strategy == "mean_reversion":
        strategy_params = {
            'rsi_oversold': st.sidebar.slider("RSI Oversold", 20, 40, 30),
            'rsi_overbought': st.sidebar.slider("RSI Overbought", 60, 80, 70),
            'volume_threshold': st.sidebar.slider("Volume Threshold", 1.0, 3.0, 1.5),
            'bb_period': st.sidebar.slider("BB Period", 10, 30, 20)
        }
    elif selected_strategy == "momentum":
        strategy_params = {
            'fast_ma': st.sidebar.slider("Fast MA", 5, 20, 10),
            'slow_ma': st.sidebar.slider("Slow MA", 30, 100, 50),
            'rsi_momentum_threshold': st.sidebar.slider("RSI Momentum", 40, 60, 50),
            'min_momentum_strength': st.sidebar.slider("Momentum Strength", 0.01, 0.05, 0.02)
        }
    elif selected_strategy == "multi_factor":
        strategy_params = {
            'technical_weight': st.sidebar.slider("Technical Weight", 0.1, 0.8, 0.4),
            'sentiment_weight': st.sidebar.slider("Sentiment Weight", 0.1, 0.8, 0.3),
            'momentum_weight': st.sidebar.slider("Momentum Weight", 0.1, 0.8, 0.3),
            'signal_threshold': st.sidebar.slider("Signal Threshold", 0.3, 0.8, 0.5)
        }
    
    # Risk Management
    st.sidebar.subheader("Risk Management")
    strategy_params.update({
        'max_position_size': st.sidebar.slider("Max Position Size", 0.05, 0.5, 0.1),
        'stop_loss_pct': st.sidebar.slider("Stop Loss %", 0.02, 0.1, 0.05),
        'take_profit_pct': st.sidebar.slider("Take Profit %", 0.05, 0.3, 0.15)
    })
    
    # Run Analysis Button
    run_analysis = st.sidebar.button("Run Analysis", type="primary")
    
    # Main content area
    if run_analysis:
        with st.spinner("Fetching data and running analysis..."):
            # Load data
            market_data, news_data, error = load_data(
                symbol, 
                datetime.combine(start_date, datetime.min.time()), 
                datetime.combine(end_date, datetime.min.time())
            )
            
            if error:
                st.error(f"Error loading data: {error}")
                return
            
            if market_data is None or market_data.empty:
                st.error("No market data available for the selected period")
                return
            
            # Process features
            processed_data, error = process_features(market_data, news_data, symbol)
            
            if error:
                st.error(f"Error processing features: {error}")
                return
            
            # Run backtest
            results, strategy, error = run_backtest(selected_strategy, strategy_params, processed_data, symbol)
            
            if error:
                st.error(f"Error running backtest: {error}")
                return
            
            # Store results in session state
            st.session_state.analysis_results = results
            st.session_state.current_data = processed_data
            st.session_state.current_strategy = strategy
            
        st.success("Analysis completed!")
        
        # Display results
        display_results(results, processed_data, strategy, symbol)
    
    elif st.session_state.analysis_results is not None:
        # Display cached results
        display_results(
            st.session_state.analysis_results,
            st.session_state.current_data,
            st.session_state.current_strategy,
            symbol
        )
    
    else:
        display_welcome_screen()

def display_results(results, data, strategy, symbol):
    """Display analysis results in tabs."""
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Trade Analysis", "AI Explanations", "Feature Importance"])
    
    with tab1:
        display_performance_tab(results, data)
    
    with tab2:
        display_trade_analysis_tab(results)
    
    with tab3:
        display_ai_explanations_tab(results, data, strategy)
    
    with tab4:
        display_feature_importance_tab(data, strategy)

def display_performance_tab(results, data):
    """Display performance metrics and charts."""
    st.subheader("Backtest Performance")
    
    # Performance metrics - fix delta calculations
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # Delta should be 0 if no trades were executed
        excess_delta = results['excess_return'] if results['total_trades'] > 0 else 0
        st.metric("Total Return", f"{results['total_return']:.2%}")
    with col2:
        # Information ratio as delta only if meaningful
        info_ratio_delta = results['information_ratio'] if results['total_trades'] > 0 else 0
        st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
    with col3:
        # Drawdown duration delta only if trades occurred
        dd_duration_delta = results['avg_drawdown_duration'] if results['total_trades'] > 0 else 0
        st.metric("Max Drawdown", f"{results['max_drawdown']:.2%}")
    with col4:
        # Trade count as delta
        st.metric("Win Rate", f"{results['win_rate']:.2%}")
    
    # Additional metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Annualized Return", f"{results['annualized_return']:.2%}")
    with col2:
        st.metric("Volatility", f"{results['volatility']:.2%}")
    with col3:
        st.metric("Profit Factor", f"{results['profit_factor']:.2f}")
    with col4:
        st.metric("Final Portfolio", f"${results['final_portfolio_value']:,.2f}")
    
    # Equity curve
    st.subheader("Equity Curve")
    equity_df = results['equity_curve']
    
    fig = go.Figure()
    
    # Portfolio value
    fig.add_trace(go.Scatter(
        x=equity_df['date'],
        y=equity_df['portfolio_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='blue', width=2)
    ))
    
    # Benchmark (buy and hold)
    if not data.empty:
        benchmark_returns = data['Close'] / data['Close'].iloc[0] * 100000
        fig.add_trace(go.Scatter(
            x=data.index,
            y=benchmark_returns,
            mode='lines',
            name='Buy & Hold',
            line=dict(color='gray', width=1, dash='dash')
        ))
    
    fig.update_layout(
        title="Portfolio Performance vs Buy & Hold",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Drawdown chart
    st.subheader("Drawdown Analysis")
    drawdown_df = results['drawdown_series']
    
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=equity_df['date'],
        y=drawdown_df * 100,
        mode='lines',
        name='Drawdown',
        fill='tozeroy',
        line=dict(color='red', width=1)
    ))
    
    fig_dd.update_layout(
        title="Portfolio Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_dd, use_container_width=True)

def display_trade_analysis_tab(results):
    """Display trade analysis and statistics."""
    st.subheader("Trade Analysis")
    
    trades_df = results['trades']
    
    if not trades_df.empty:
        # Trade statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Trade Statistics")
            st.write(f"**Total Trades:** {results['total_trades']}")
            st.write(f"**Winning Trades:** {results['winning_trades']}")
            st.write(f"**Losing Trades:** {results['losing_trades']}")
            st.write(f"**Win Rate:** {results['win_rate']:.2%}")
            st.write(f"**Average Win:** ${results['avg_win']:.2f}")
            st.write(f"**Average Loss:** ${results['avg_loss']:.2f}")
        
        with col2:
            st.subheader("Cost Analysis")
            st.write(f"**Total Commission:** ${results['total_commission']:.2f}")
            st.write(f"**Total Slippage:** ${results['total_slippage']:.2f}")
            st.write(f"**Commission %:** {results['commission_pct']:.3%}")
            st.write(f"**Slippage %:** {results['slippage_pct']:.3%}")
        
        # Trade log
        st.subheader("Trade Log")
        display_df = trades_df.copy()
        if 'date' in display_df.columns:
            display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d')
        
        st.dataframe(display_df, use_container_width=True)
        
        # P&L distribution
        if 'pnl' in trades_df.columns:
            st.subheader("P&L Distribution")
            pnl_data = trades_df['pnl'].dropna()
            
            fig = px.histogram(
                x=pnl_data,
                nbins=20,
                title="Trade P&L Distribution"
            )
            fig.update_layout(
                xaxis_title="P&L ($)",
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("No trades were executed during the backtest period")

def display_ai_explanations_tab(results, data, strategy):
    """Display AI explanations for trading decisions."""
    st.subheader("AI Decision Explanations")
    st.markdown("*Understanding why the AI made each trading decision*")
    
    trades_df = results['trades']
    
    if not trades_df.empty:
        # Trade selection for explanation
        trade_options = []
        for _, trade in trades_df.iterrows():
            date_str = pd.to_datetime(trade['date']).strftime('%Y-%m-%d')
            trade_options.append(f"{trade['action']} on {date_str} at ${trade['price']:.2f}")
        
        selected_trade_idx = st.selectbox(
            "Select Trade to Explain",
            range(len(trade_options)),
            format_func=lambda x: trade_options[x]
        )
        
        if selected_trade_idx is not None:
            selected_trade = trades_df.iloc[selected_trade_idx]
            trade_date = pd.to_datetime(selected_trade['date'])
            signal = 1 if selected_trade['action'] == 'BUY' else -1
            
            # Generate explanation
            try:
                explainer = Explainer()
                explanation = explainer.explain_trade_decision(
                    strategy, data, trade_date, signal
                )
                
                display_trade_explanation(explanation, selected_trade)
                
            except Exception as e:
                st.error(f"Error generating explanation: {e}")
    
    else:
        st.warning("No trades available for explanation")

def display_trade_explanation(explanation, trade):
    """Display detailed trade explanation."""
    
    # Trade summary
    st.subheader("Trade Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Action:** {trade['action']}")
        st.write(f"**Price:** ${trade['price']:.2f}")
    with col2:
        st.write(f"**Shares:** {trade['shares']}")
        st.write(f"**Value:** ${trade['value']:.2f}")
    with col3:
        if 'pnl' in trade.index:
            st.write(f"**P&L:** ${trade['pnl']:.2f}")
            st.write(f"**P&L %:** {trade['pnl_pct']:.2%}")
    
    # Decision explanation
    st.subheader("Decision Explanation")
    st.write(explanation.get('signal_description', 'No description available'))
    
    # Strategy-specific explanations
    if explanation.get('explanation_type') == 'rule_based':
        # Rule-based explanation
        if 'triggered_conditions' in explanation:
            st.subheader("Triggered Conditions")
            for condition in explanation['triggered_conditions']:
                st.write(f"✓ {condition}")
        
        if 'decision_rationale' in explanation:
            st.subheader("Decision Rationale")
            for rationale in explanation['decision_rationale']:
                st.write(f"• {rationale}")
        
        if 'feature_analysis' in explanation:
            st.subheader("Feature Analysis")
            for feature, analysis in explanation['feature_analysis'].items():
                st.write(f"**{feature.replace('_', ' ').title()}:**")
                for key, value in analysis.items():
                    st.write(f"  - {key}: {value}")
    
    elif explanation.get('explanation_type') == 'ml_based':
        # ML-based explanation with SHAP values
        if 'feature_importance' in explanation:
            st.subheader("Feature Importance")
            
            # Create DataFrame for visualization
            importance_df = pd.DataFrame(explanation['feature_importance'])
            
            # Feature importance chart
            fig = px.bar(
                importance_df.head(10),
                x='abs_importance',
                y='feature',
                orientation='h',
                color='contribution',
                title="Top 10 Feature Contributions",
                color_discrete_map={'positive': 'green', 'negative': 'red'}
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Top factors
            if 'top_positive_factors' in explanation:
                st.subheader("Top Positive Factors")
                for factor in explanation['top_positive_factors']:
                    st.write(f"✓ **{factor['feature']}**: {factor['description']}")
            
            if 'top_negative_factors' in explanation:
                st.subheader("Top Negative Factors")
                for factor in explanation['top_negative_factors']:
                    st.write(f"✗ **{factor['feature']}**: {factor['description']}")
        
        if 'confidence_score' in explanation:
            st.subheader("Confidence Score")
            st.progress(explanation['confidence_score'])
            st.write(f"Model confidence: {explanation['confidence_score']:.2%}")
    
    # Risk factors
    if 'risk_factors' in explanation:
        st.subheader("Risk Factors")
        for risk in explanation['risk_factors']:
            st.warning(f"{risk}")
    
    # Market context
    if 'market_context' in explanation:
        st.subheader("Market Context")
        context = explanation['market_context']
        for key, value in context.items():
            st.write(f"**{key.replace('_', ' ').title()}:** {value}")

def display_feature_importance_tab(data, strategy):
    """Display global feature importance analysis."""
    st.subheader("Global Feature Importance")
    st.markdown("*Most influential factors across all trading decisions*")
    
    # Calculate feature importance based on available data
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) > 0:
        # Calculate correlation with returns
        if 'Close' in data.columns:
            returns = data['Close'].pct_change().dropna()
            correlations = data[numeric_columns].corrwith(returns).abs().sort_values(ascending=False)
            
            # Create importance visualization
            importance_df = pd.DataFrame({
                'Feature': correlations.index,
                'Importance': correlations.values
            }).head(15)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Feature Importance (Correlation with Returns)"
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature descriptions
            st.subheader("Feature Descriptions")
            for feature in importance_df['Feature'].head(10):
                description = get_feature_description(feature)
                st.write(f"**{feature}:** {description}")
    
    else:
        st.warning("No numeric features available for analysis")

def get_feature_description(feature):
    """Get description for a feature."""
    descriptions = {
        'RSI': 'Relative Strength Index - measures overbought/oversold conditions',
        'MACD': 'Moving Average Convergence Divergence - trend and momentum indicator',
        'BB_Position': 'Bollinger Bands Position - price position within bands',
        'Volume_Ratio': 'Volume Ratio - current volume vs. average volume',
        'Price_Change': 'Price Change - recent price movement',
        'sentiment_mean': 'News Sentiment - average sentiment from news analysis',
        'SMA_10': '10-day Simple Moving Average',
        'SMA_50': '50-day Simple Moving Average',
        'Volatility_20': '20-day Price Volatility',
        'Close': 'Closing Price',
        'Volume': 'Trading Volume'
    }
    
    return descriptions.get(feature, f'{feature} - technical indicator')

def display_welcome_screen():
    """Display welcome screen when no analysis is run."""
    st.markdown("## Welcome to Aura!")
    st.markdown("""
    **Aura** is your transparent AI trading companion that explains the 'why' behind every trade decision.
    
    ### How to use:
    1. **Select an asset** in the sidebar (e.g., AAPL, GOOGL)
    2. **Choose your date range** for backtesting
    3. **Pick a trading strategy** from our library
    4. **Adjust parameters** to customize the strategy
    5. **Click 'Run Analysis'** to see the magic happen!
    
    ### What makes Aura special:
    - **Explainable AI**: See exactly why each trade was made
    - **Visual Insights**: Interactive charts and feature analysis
    - **Educational**: Perfect for learning quantitative trading
    - **Transparent**: No more black-box trading algorithms
    
    ### Available Strategies:
    - **Simple MA Crossover**: Basic moving average crossover strategy - perfect for beginners
    - **Mean Reversion**: Buys oversold assets, sells overbought ones
    - **Momentum**: Follows trending price movements
    - **Multi-Factor**: Combines technical and sentiment analysis
    
    Ready to start? Configure your analysis in the sidebar and click **Run Analysis**!
    """)

if __name__ == "__main__":
    main()