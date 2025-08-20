import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import sys
import os
import pandas as pd
import numpy as np

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
if backend_path not in sys.path:
    sys.path.append(backend_path)

# Initialize backend availability flag
BACKEND_AVAILABLE = False

try:
    from backend.data_manager import DataManager
    from backend.feature_engine import FeatureEngine
    from backend.strategies import BaseStrategy
    from backend.backtester import BacktestEngine
    from backend.explainer import TradingStrategyExplainer
    BACKEND_AVAILABLE = True
except ImportError as e:
    st.error(f"Backend import error: {str(e)}")
    st.error("Please ensure all backend files are properly installed:")
    st.code("""
    Required files:
    - backend/data_manager.py
    - backend/feature_engine.py  
    - backend/strategies.py
    - backend/backtester.py
    - backend/explainer.py
    """)
    st.stop()

warnings.filterwarnings('ignore')

def load_data(symbol, start_date, end_date):
    """Load market data with enhanced error handling."""
    try:
        # Validate inputs
        if not symbol or symbol.strip() == "":
            raise ValueError("Symbol cannot be empty")
        
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")
        
        # Convert to datetime if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        dm = DataManager()
        
        # Validate symbol first
        if not dm.validate_symbol(symbol):
            raise ValueError(f"Invalid or unknown symbol: {symbol}")
        
        market_data = dm.fetch_market_data(symbol, start_date, end_date)
        
        if market_data is None or market_data.empty:
            raise Exception(f"No data retrieved for {symbol} between {start_date} and {end_date}")
        
        # Basic data validation
        if len(market_data) < 10:
            raise ValueError(f"Insufficient data: only {len(market_data)} records found")
        
        return market_data, None
    except Exception as e:
        error_msg = f"Data loading error: {str(e)}"
        return None, error_msg

def process_features(market_data, strategy_params):
    """Process features with enhanced error handling."""
    try:
        if market_data is None or market_data.empty:
            raise ValueError("No market data provided for feature processing")
        
        fe = FeatureEngine()
        technical_data = fe.calculate_technical_indicators(market_data, strategy_params)
        
        if technical_data is None or technical_data.empty:
            raise Exception("Feature calculation returned empty data")
        
        # Validate feature data
        if len(technical_data) < len(market_data) * 0.8:  # Allow some data loss due to indicators
            st.warning(f"Feature processing reduced data from {len(market_data)} to {len(technical_data)} rows")
        
        return technical_data, None
    except Exception as e:
        error_msg = f"Feature processing error: {str(e)}"
        return None, error_msg

def run_backtest(strategy_name, strategy_params, data, symbol):
    """Run backtest with enhanced error handling."""
    try:
        if data is None or data.empty:
            raise ValueError("No data provided for backtesting")
        
        # Create strategy using the factory method
        strategy = BaseStrategy.create_strategy(strategy_name, strategy_params)
        
        # Initialize backtester with default parameters
        backtester = BacktestEngine(
            initial_capital=100000, 
            commission_rate=0.001, 
            slippage_rate=0.0005
        )
        
        # Add debugging for LSTM strategies
        if strategy_name == "lstm_strategy":
            st.info(f"🧠 Running LSTM strategy with {len(data)} data points and {len(data.columns)} features")
            
            # Check if we have enough data for LSTM
            min_required = strategy_params.get('min_data_length', 300)
            if len(data) < min_required:
                raise ValueError(f"LSTM requires at least {min_required} data points, got {len(data)}")
        
        # Run backtest
        with st.spinner(f"Running {strategy_name.replace('_', ' ').title()} backtest..."):
            results = backtester.run_backtest(strategy, data, symbol)
        
        if results is None:
            raise Exception("Backtest returned no results")
        
        # Check for backtest errors
        if 'error' in results:
            raise Exception(results['error'])
        
        # Store strategy instance for explainer usage
        results['strategy_instance'] = strategy
        
        return results, None
    except Exception as e:
        import traceback
        error_msg = f"Backtest error: {str(e)}"
        print(f"Full error: {traceback.format_exc()}")
        return None, error_msg

def validate_strategy_params(strategy_name, params):
    """Validate strategy parameters."""
    try:
        if strategy_name == "simple_ma":
            if params['fast_ma'] >= params['slow_ma']:
                return False, "Fast MA period must be less than Slow MA period"
        
        elif strategy_name == "mean_reversion":
            if params['rsi_oversold'] >= params['rsi_overbought']:
                return False, "RSI Oversold threshold must be less than Overbought threshold"
        
        elif strategy_name == "momentum":
            if params['fast_ma'] >= params['slow_ma']:
                return False, "Fast MA period must be less than Slow MA period"
        
        elif strategy_name == "lstm_strategy":
            if params['lookback_window'] < 10:
                return False, "Lookback window must be at least 10 days"
            if params['prediction_horizon'] < 1:
                return False, "Prediction horizon must be at least 1 day"
        
        return True, ""
    except Exception as e:
        return False, f"Parameter validation error: {str(e)}"

def main():
    try:
        st.set_page_config(page_title="Aura - Explainable AI Trading Platform", page_icon="📈", layout="wide")
        
        st.title("📈 Aura - Explainable AI Trading Platform")
        st.markdown("_*A comprehensive platform to backtest AI-powered trading strategies with explainable AI.*_")
        
        # Check backend availability
        if not BACKEND_AVAILABLE:
            st.error("❌ Backend modules are not available. Please ensure all backend files are in the correct directory.")
            st.stop()
        
        # Sidebar configuration
        st.sidebar.header("📋 Configuration")
        
        # Stock selection with more options
        stock_options = {
            "Apple": "AAPL", 
            "Google": "GOOGL", 
            "Microsoft": "MSFT", 
            "Amazon": "AMZN", 
            "Tesla": "TSLA",
            "Netflix": "NFLX",
            "Meta": "META",
            "NVIDIA": "NVDA",
            "SPY ETF": "SPY"
        }
        selected_stock = st.sidebar.selectbox("Select Stock", list(stock_options.keys()))
        symbol = stock_options[selected_stock]
        
        # Date range selection
        st.sidebar.subheader("📅 Date Range")
        max_date = datetime.now().date()
        default_start = max_date - timedelta(days=730)  # 2 years for LSTM
        
        start_date = st.sidebar.date_input("Start Date", default_start, max_value=max_date)
        end_date = st.sidebar.date_input("End Date", max_date, max_value=max_date)
        
        # Validate date range
        if start_date >= end_date:
            st.sidebar.error("⚠️ Start date must be before end date")
            st.stop()
        
        # Calculate data range info
        date_range_days = (end_date - start_date).days
        st.sidebar.info(f"📊 Date range: {date_range_days} days")
        
        # Strategy selection
        st.sidebar.subheader("🎯 Strategy Selection")
        strategy_options = {
            "Simple MA Crossover": "simple_ma", 
            "Mean Reversion": "mean_reversion", 
            "Momentum": "momentum", 
            "LSTM Deep Learning": "lstm_strategy"
        }
        selected_strategy_name = st.sidebar.selectbox("Choose Strategy", list(strategy_options.keys()))
        selected_strategy = strategy_options[selected_strategy_name]
        
        # Strategy parameter configuration
        st.sidebar.subheader("⚙️ Strategy Parameters")
        strategy_params = {}
        
        if selected_strategy == "simple_ma":
            st.sidebar.markdown("**Moving Average Crossover Settings**")
            fast_ma = st.sidebar.slider("Fast MA Period", 5, 50, 10)
            slow_ma = st.sidebar.slider("Slow MA Period", 20, 200, 50)
            strategy_params = {'fast_ma': fast_ma, 'slow_ma': slow_ma}
            
        elif selected_strategy == "mean_reversion":
            st.sidebar.markdown("**Mean Reversion Settings**")
            strategy_params = {
                'bb_period': st.sidebar.slider("Bollinger Bands Period", 10, 50, 20),
                'rsi_period': st.sidebar.slider("RSI Period", 5, 30, 14),
                'rsi_oversold': st.sidebar.slider("RSI Oversold Threshold", 20, 40, 30),
                'rsi_overbought': st.sidebar.slider("RSI Overbought Threshold", 60, 80, 70),
                'rsi_neutral_upper': st.sidebar.slider("RSI Neutral Upper", 50, 65, 55),
                'rsi_neutral_lower': st.sidebar.slider("RSI Neutral Lower", 35, 50, 45)
            }
                
        elif selected_strategy == "momentum":
            st.sidebar.markdown("**Momentum Strategy Settings**")
            fast_ma = st.sidebar.slider("Fast MA", 5, 50, 12)
            slow_ma = st.sidebar.slider("Slow MA", 20, 200, 26)
            strategy_params = {
                'fast_ma': fast_ma,
                'slow_ma': slow_ma,
                'rsi_momentum_threshold': st.sidebar.slider("RSI Momentum Threshold", 40, 60, 50),
                'rsi_strong_momentum': st.sidebar.slider("RSI Strong Momentum", 55, 75, 60),
                'rsi_weak_momentum': st.sidebar.slider("RSI Weak Momentum", 25, 45, 40),
                'macd_strength_threshold': st.sidebar.slider("MACD Strength", 0.0001, 0.01, 0.001, format="%.4f")
            }
            
        elif selected_strategy == "lstm_strategy":
            st.sidebar.markdown("**🧠 LSTM Deep Learning Settings**")
            
            # Check if we have enough data for LSTM
            min_data_required = 300
            if date_range_days < min_data_required:
                st.sidebar.warning(f"⚠️ LSTM requires at least {min_data_required} days of data. Current range: {date_range_days} days")
            
            # Quick test configuration button
            st.sidebar.markdown("---")
            if st.sidebar.button("🚀 Use Quick Test Config", help="Use optimized settings for testing"):
                st.sidebar.success("✅ Quick test configuration applied!")
                # Set optimal testing parameters
                lookback_window = 30
                prediction_horizon = 1
                lstm_layer1 = 32
                lstm_layer2 = 16
                dropout_rate = 0.2
                learning_rate = 0.001
                batch_size = 32
                epochs = 25  # Reduced for faster testing
                early_stopping_patience = 8
                signal_threshold = 0.4  # Lower threshold for more signals
                max_position_size = 0.1
                stop_loss_pct = 0.05
                take_profit_pct = 0.10
                retrain_frequency = 30
                use_shap = True
                max_features = 15
            else:
                # Regular configuration
                with st.sidebar.expander("Model Architecture", expanded=True):
                    lookback_window = st.slider("Lookback Window (Days)", 30, 120, 60)
                    prediction_horizon = st.slider("Prediction Horizon (Days)", 1, 10, 5)
                    lstm_layer1 = st.slider("LSTM Layer 1 Units", 32, 128, 64)
                    lstm_layer2 = st.slider("LSTM Layer 2 Units", 16, 64, 32)
                    dropout_rate = st.slider("Dropout Rate", 0.1, 0.5, 0.2)
                
                with st.sidebar.expander("Training Parameters"):
                    learning_rate = st.select_slider("Learning Rate", 
                                                   options=[0.0001, 0.0005, 0.001, 0.005, 0.01], 
                                                   value=0.001)
                    batch_size = st.select_slider("Batch Size", 
                                                options=[16, 32, 64, 128], 
                                                value=32)
                    epochs = st.slider("Max Epochs", 50, 200, 100)
                    early_stopping_patience = st.slider("Early Stopping Patience", 5, 20, 15)
                
                with st.sidebar.expander("Signal & Risk Parameters"):
                    signal_threshold = st.slider("Signal Confidence Threshold", 0.3, 0.9, 0.6)
                    max_position_size = st.slider("Max Position Size", 0.05, 0.25, 0.1)
                    stop_loss_pct = st.slider("Stop Loss %", 0.02, 0.10, 0.05)
                    take_profit_pct = st.slider("Take Profit %", 0.05, 0.25, 0.15)
                
                with st.sidebar.expander("Advanced Settings"):
                    retrain_frequency = st.slider("Retrain Frequency (Days)", 7, 60, 30)
                    use_shap = st.checkbox("Enable SHAP Explanations", value=True)
                    max_features = st.slider("Max Features for SHAP", 10, 30, 20)
            
            # Add helpful info about signal threshold
            st.sidebar.info(f"💡 Signal Threshold: {signal_threshold:.1f} - Lower values generate more signals")
            
            strategy_params = {
                'lookback_window': lookback_window,
                'prediction_horizon': prediction_horizon,
                'lstm_units': [lstm_layer1, lstm_layer2],
                'dropout_rate': dropout_rate,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'epochs': epochs,
                'early_stopping_patience': early_stopping_patience,
                'signal_threshold': signal_threshold,
                'retrain_frequency': retrain_frequency,
                'min_data_length': min_data_required,
                'use_shap': use_shap,
                'max_features': max_features,
                'max_position_size': max_position_size,
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct
            }
        
        # Validate parameters
        params_valid, param_error = validate_strategy_params(selected_strategy, strategy_params)
        if not params_valid:
            st.sidebar.error(f"⚠️ {param_error}")
            st.stop()
        
        # Run analysis button
        st.sidebar.markdown("---")
        run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        if run_analysis:
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Load data
                status_text.text("📡 Loading market data...")
                progress_bar.progress(25)
                
                market_data, error = load_data(symbol, start_date, end_date)
                if error:
                    st.error(f"❌ Data loading failed: {error}")
                    st.stop()
                
                st.success(f"✅ Loaded {len(market_data)} data points for {symbol}")
                
                # Step 2: Process features
                status_text.text("🔧 Processing technical indicators...")
                progress_bar.progress(50)
                
                processed_data, error = process_features(market_data, strategy_params)
                if error:
                    st.error(f"❌ Feature processing failed: {error}")
                    st.stop()
                
                # Step 3: Run backtest
                status_text.text("📊 Running backtest...")
                progress_bar.progress(75)
                
                results, error = run_backtest(selected_strategy, strategy_params, processed_data, symbol)
                if error:
                    st.error(f"❌ Backtest failed: {error}")
                    st.stop()
                
                # Step 4: Complete
                status_text.text("✅ Analysis complete!")
                progress_bar.progress(100)
                
                # Store results in session state for explainer
                st.session_state['results'] = results
                st.session_state['processed_data'] = processed_data
                st.session_state['strategy_type'] = selected_strategy
                st.session_state['symbol'] = symbol
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Display results
                display_results(results, processed_data, selected_strategy)
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Unexpected error: {str(e)}")
    
    except Exception as e:
        st.error(f"❌ Critical error in main application: {str(e)}")
        import traceback
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())

def display_results(results, data, strategy_type):
    # Create tabs based on strategy type
    if strategy_type == "lstm_strategy":
        tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Trade Analysis", "AI Explanations", "Strategy Insights"])
    else:
        tab1, tab2, tab3 = st.tabs(["Performance", "Trade Analysis", "Strategy Insights"])
    
    with tab1:
        display_performance(results, data)
    with tab2:
        display_trade_analysis(results)
    if strategy_type == "lstm_strategy":
        with tab3:
            display_ai_explanations(results, data)
        with tab4:
            display_strategy_insights(results, strategy_type)
    else:
        with tab3:
            display_strategy_insights(results, strategy_type)

def display_performance(results, data):
    """Enhanced performance display with better error handling."""
    try:
        st.subheader("📊 Backtest Performance")
        
        # Safely get metrics with defaults
        total_return = results.get('total_return', 0)
        sharpe_ratio = results.get('sharpe_ratio', 0)
        max_drawdown = results.get('max_drawdown', 0)
        win_rate = results.get('win_rate', 0)
        annualized_return = results.get('annualized_return', 0)
        volatility = results.get('volatility', 0)
        profit_factor = results.get('profit_factor', 0)
        final_value = results.get('final_portfolio_value', 100000)
        
        # Performance metrics grid
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Return", f"{total_return:.2%}", 
                    delta=f"{total_return:.2%}" if total_return != 0 else None)
        col2.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}",
                    delta="Good" if sharpe_ratio > 1 else "Poor" if sharpe_ratio < 0 else "Average")
        col3.metric("Max Drawdown", f"{max_drawdown:.2%}",
                    delta="Low" if abs(max_drawdown) < 0.1 else "High")
        col4.metric("Win Rate", f"{win_rate:.1%}",
                    delta="Good" if win_rate > 0.5 else "Poor")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Annualized Return", f"{annualized_return:.2%}")
        col2.metric("Volatility", f"{volatility:.2%}")
        col3.metric("Profit Factor", f"{profit_factor:.2f}")
        col4.metric("Final Portfolio", f"${final_value:,.2f}")

        # Equity curve with buy & hold comparison
        st.subheader("📈 Equity Curve")
        equity_curve = results.get('equity_curve', pd.Series())
        if not equity_curve.empty and not data.empty:
            fig = go.Figure()
            
            # Strategy performance
            fig.add_trace(go.Scatter(
                x=equity_curve.index, 
                y=equity_curve, 
                mode='lines', 
                name='Strategy',
                line=dict(color='blue', width=2)
            ))
            
            # Buy & hold benchmark
            if 'Close' in data.columns:
                initial_value = 100000
                buy_hold = data['Close'] / data['Close'].iloc[0] * initial_value
                fig.add_trace(go.Scatter(
                    x=data.index, 
                    y=buy_hold, 
                    mode='lines', 
                    name='Buy & Hold',
                    line=dict(color='red', width=2, dash='dash')
                ))
            
            fig.update_layout(
                title='Portfolio Value Over Time',
                xaxis_title='Date',
                yaxis_title='Portfolio Value ($)',
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No equity curve data available")

        # Drawdown analysis
        st.subheader("📉 Drawdown Analysis")
        drawdown_series = results.get('drawdown_series', pd.Series())
        if not drawdown_series.empty:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=drawdown_series.index, 
                y=drawdown_series * 100, 
                mode='lines', 
                name='Drawdown',
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.3)',
                line=dict(color='red')
            ))
            fig_dd.update_layout(
                title='Portfolio Drawdown',
                xaxis_title='Date',
                yaxis_title='Drawdown (%)',
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig_dd, use_container_width=True)
        else:
            st.warning("No drawdown data available")
    except Exception as e:
        st.error(f"Error displaying performance: {str(e)}")

def display_trade_analysis(results):
    """Enhanced trade analysis with better formatting."""
    try:
        st.subheader("💼 Trade Analysis")
        
        trades_df = results.get('trades', pd.DataFrame())
        if trades_df.empty:
            st.warning("⚠️ No trades were executed during the backtest period.")
            st.info("""
            Possible reasons:
            - Strategy parameters are too conservative
            - Market conditions didn't meet strategy criteria
            - Insufficient data for signal generation
            """)
            return
        
        # Trade statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Trade Statistics")
            metrics = [
                ("Total Trades", results.get('total_trades', 0)),
                ("Winning Trades", results.get('winning_trades', 0)),
                ("Losing Trades", results.get('losing_trades', 0)),
                ("Win Rate", f"{results.get('win_rate', 0):.1%}"),
                ("Average Win", f"${results.get('avg_win', 0):.2f}"),
                ("Average Loss", f"${results.get('avg_loss', 0):.2f}")
            ]
            
            for metric, value in metrics:
                st.write(f"**{metric}:** {value}")
        
        with col2:
            st.markdown("### 💰 Cost Analysis")
            cost_metrics = [
                ("Total Commission", f"${results.get('total_commission', 0):.2f}"),
                ("Total Slippage", f"${results.get('total_slippage', 0):.2f}"),
                ("Commission %", f"{results.get('commission_pct', 0):.3%}"),
                ("Slippage %", f"{results.get('slippage_pct', 0):.3%}")
            ]
            
            for metric, value in cost_metrics:
                st.write(f"**{metric}:** {value}")

        # Trade log
        st.subheader("📋 Trade Log")
        if len(trades_df) > 0:
            # Format the trades dataframe for better display
            display_df = trades_df.copy()
            
            # Format columns if they exist
            if 'date' in display_df.columns:
                display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d')
            if 'price' in display_df.columns:
                display_df['price'] = display_df['price'].round(2)
            if 'value' in display_df.columns:
                display_df['value'] = display_df['value'].round(2)
            if 'pnl' in display_df.columns:
                display_df['pnl'] = display_df['pnl'].round(2)
            
            st.dataframe(display_df, use_container_width=True)

            # P&L distribution
            if 'pnl' in trades_df.columns:
                pnl_data = trades_df['pnl'].dropna()
                if not pnl_data.empty:
                    st.subheader("📊 P&L Distribution")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.histogram(
                            pnl_data, 
                            nbins=min(20, len(pnl_data)), 
                            title="Trade P&L Distribution"
                        )
                        fig.update_layout(
                            xaxis_title="P&L ($)",
                            yaxis_title="Number of Trades",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # P&L statistics
                        st.markdown("### P&L Statistics")
                        st.write(f"**Mean P&L:** ${pnl_data.mean():.2f}")
                        st.write(f"**Median P&L:** ${pnl_data.median():.2f}")
                        st.write(f"**Std Dev:** ${pnl_data.std():.2f}")
                        st.write(f"**Best Trade:** ${pnl_data.max():.2f}")
                        st.write(f"**Worst Trade:** ${pnl_data.min():.2f}")
    except Exception as e:
        st.error(f"Error displaying trade analysis: {str(e)}")

def display_ai_explanations(results, data):
    """Enhanced AI explanations with better error handling."""
    try:
        st.subheader("🧠 AI Model Explanations")
        
        strategy_instance = results.get('strategy_instance')
        if not strategy_instance:
            st.warning("⚠️ Strategy instance not available for explanations.")
            return
        
        # Check if strategy supports explanations
        if not hasattr(strategy_instance, 'get_feature_importance'):
            st.info("ℹ️ This strategy does not support AI explanations.")
            return
        
        # Initialize explainer
        explainer = TradingStrategyExplainer(strategy_instance, data)
        
        # Check for available explanations
        available_dates = []
        if hasattr(strategy_instance, 'explanation_cache') and strategy_instance.explanation_cache:
            available_dates = list(strategy_instance.explanation_cache.keys())
            available_dates.sort(reverse=True)  # Most recent first
        
        if available_dates:
            # Date selection
            st.markdown("### 📅 Select Analysis Date")
            col1, col2 = st.columns([3, 1])
            
            with col1:
                selected_date = st.selectbox(
                    "Choose a prediction date to analyze:",
                    available_dates,
                    format_func=lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if hasattr(x, 'strftime') else str(x)
                )
            
            with col2:
                st.metric("Available Dates", len(available_dates))
            
            # Generate comprehensive explanation
            with st.spinner("🔄 Generating AI explanations..."):
                explanation = explainer.generate_comprehensive_explanation(selected_date)
            
            if "error" in explanation:
                st.error(f"❌ {explanation['error']}")
                return
            
            # Display narrative explanation
            if 'narrative' in explanation and explanation['narrative']:
                st.markdown("### 📝 Model Reasoning")
                st.markdown(explanation['narrative'])
            
            # Display feature importance
            if 'feature_importance' in explanation and explanation['feature_importance']:
                st.markdown("### 🎯 Feature Importance")
                feature_imp = explanation['feature_importance']
                
                # Create feature importance chart
                top_features = dict(list(feature_imp.items())[:10])
                if top_features:
                    fig_features = go.Figure(data=[
                        go.Bar(
                            y=list(top_features.keys()),
                            x=list(top_features.values()),
                            orientation='h',
                            marker_color='steelblue',
                            text=[f"{v:.1%}" for v in top_features.values()],
                            textposition='auto'
                        )
                    ])
                    fig_features.update_layout(
                        title='Top 10 Most Important Features',
                        xaxis_title='Importance Score',
                        yaxis_title='Features',
                        height=500,
                        margin=dict(l=150)  # More space for feature names
                    )
                    st.plotly_chart(fig_features, use_container_width=True)
                    
                    # Feature importance table
                    with st.expander("📊 Detailed Feature Importance"):
                        importance_df = pd.DataFrame([
                            {'Feature': k, 'Importance': f"{v:.3%}", 'Score': v}
                            for k, v in feature_imp.items()
                        ])
                        st.dataframe(importance_df, use_container_width=True)
            
            # Display market context
            if 'market_context' in explanation and explanation['market_context']:
                st.markdown("### 📊 Market Context")
                market_ctx = explanation['market_context']
                
                if 'error' not in market_ctx:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if 'current_price' in market_ctx:
                            st.metric("Current Price", f"${market_ctx['current_price']:.2f}")
                    with col2:
                        if 'trend_30d' in market_ctx:
                            trend_value = market_ctx['trend_30d']
                            st.metric("30-Day Trend", f"{trend_value:.1%}", 
                                    delta="Bullish" if trend_value > 0 else "Bearish")
                    with col3:
                        if 'volatility_annualized' in market_ctx:
                            vol_value = market_ctx['volatility_annualized']
                            st.metric("Volatility (Ann.)", f"{vol_value:.1%}",
                                    delta="High" if vol_value > 0.3 else "Low" if vol_value < 0.15 else "Normal")
                    with col4:
                        if 'price_position' in market_ctx:
                            pos_value = market_ctx['price_position']
                            st.metric("Price Position", f"{pos_value:.1%}",
                                    delta="High" if pos_value > 0.8 else "Low" if pos_value < 0.2 else "Mid")
            
            # Display confidence metrics
            if 'confidence_metrics' in explanation and explanation['confidence_metrics']:
                st.markdown("### 🎯 Prediction Confidence")
                conf_metrics = explanation['confidence_metrics']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if 'shap_confidence' in conf_metrics:
                        shap_conf = conf_metrics['shap_confidence']
                        st.metric("SHAP Confidence", f"{shap_conf:.1%}",
                                delta="High" if shap_conf > 0.7 else "Low" if shap_conf < 0.4 else "Medium")
                with col2:
                    if 'feature_consistency' in conf_metrics:
                        feat_conf = conf_metrics['feature_consistency']
                        st.metric("Feature Consistency", f"{feat_conf:.1%}",
                                delta="Good" if feat_conf > 0.6 else "Poor")
                with col3:
                    if 'overall_confidence' in conf_metrics:
                        overall_conf = conf_metrics['overall_confidence']
                        st.metric("Overall Confidence", f"{overall_conf:.1%}",
                                delta="High" if overall_conf > 0.7 else "Low" if overall_conf < 0.4 else "Medium")
        
        else:
            st.info("ℹ️ No explanations available yet. The model needs to make predictions first.")
            
            # Display general feature importance if available
            try:
                feature_importance = strategy_instance.get_feature_importance()
                if feature_importance:
                    st.markdown("### 🎯 General Feature Importance")
                    top_features = dict(list(feature_importance.items())[:10])
                    
                    if top_features:
                        fig_features = go.Figure(data=[
                            go.Bar(
                                y=list(top_features.keys()),
                                x=list(top_features.values()),
                                orientation='h',
                                marker_color='lightcoral',
                                text=[f"{v:.1%}" for v in top_features.values()],
                                textposition='auto'
                            )
                        ])
                        fig_features.update_layout(
                            title='Top 10 Most Important Features (Overall)',
                            xaxis_title='Importance Score',
                            yaxis_title='Features',
                            height=500,
                            margin=dict(l=150)
                        )
                        st.plotly_chart(fig_features, use_container_width=True)
                        
                        # Show feature importance table
                        with st.expander("📊 Detailed Feature Importance"):
                            importance_df = pd.DataFrame([
                                {'Feature': k, 'Importance': f"{v:.3%}", 'Score': v}
                                for k, v in feature_importance.items()
                            ])
                            st.dataframe(importance_df, use_container_width=True)
                else:
                    st.warning("⚠️ No feature importance data available.")
            except Exception as e:
                st.warning(f"⚠️ Could not display general feature importance: {str(e)}")
            
            # Show model information if available
            if hasattr(strategy_instance, 'params'):
                st.markdown("### 🔧 Model Configuration")
                model_params = strategy_instance.params
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Architecture:**")
                    st.write(f"- Lookback Window: {model_params.get('lookback_window', 'N/A')} days")
                    st.write(f"- LSTM Units: {model_params.get('lstm_units', 'N/A')}")
                    st.write(f"- Dropout Rate: {model_params.get('dropout_rate', 'N/A')}")
                
                with col2:
                    st.write("**Training:**")
                    st.write(f"- Learning Rate: {model_params.get('learning_rate', 'N/A')}")
                    st.write(f"- Batch Size: {model_params.get('batch_size', 'N/A')}")
                    st.write(f"- Max Epochs: {model_params.get('epochs', 'N/A')}")
            
            # Show training status if available
            if hasattr(strategy_instance, 'last_training_date') and strategy_instance.last_training_date:
                st.info(f"🕒 Last model training: {strategy_instance.last_training_date.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.warning("⚠️ Model has not been trained yet.")
    
    except Exception as e:
        st.error(f"❌ Error generating explanations: {str(e)}")
        with st.expander("🔍 Error Details"):
            import traceback
            st.code(traceback.format_exc())

def display_strategy_insights(results, strategy_type):
    """Display strategy-specific insights and analysis."""
    try:
        st.subheader("📊 Strategy Insights")
        
        if strategy_type == "lstm_strategy":
            st.markdown("""
            ### 🧠 LSTM Deep Learning Strategy Analysis
            
            The LSTM (Long Short-Term Memory) strategy uses deep learning to:
            - **Analyze sequential patterns** in price and technical indicator data
            - **Predict future price movements** using multi-class classification
            - **Adapt to market conditions** through periodic retraining
            - **Provide explainable predictions** using SHAP values
            
            #### Key Advantages:
            - Captures complex temporal dependencies
            - Handles non-linear relationships
            - Provides prediction confidence scores
            - Offers detailed explanations for each decision
            """)
            
            # Display model-specific metrics if available
            strategy_instance = results.get('strategy_instance')
            if strategy_instance and hasattr(strategy_instance, 'last_training_date'):
                col1, col2 = st.columns(2)
                with col1:
                    if strategy_instance.last_training_date:
                        st.info(f"**Last Model Training:** {strategy_instance.last_training_date.strftime('%Y-%m-%d')}")
                    if hasattr(strategy_instance, 'params'):
                        st.info(f"**Lookback Window:** {strategy_instance.params.get('lookback_window', 'N/A')} days")
                with col2:
                    if hasattr(strategy_instance, 'params'):
                        st.info(f"**Prediction Horizon:** {strategy_instance.params.get('prediction_horizon', 'N/A')} days")
                        st.info(f"**Signal Threshold:** {strategy_instance.params.get('signal_threshold', 'N/A')}")
        
        else:
            st.markdown(f"""
            ### 📈 {strategy_type.replace('_', ' ').title()} Strategy Analysis
            
            This traditional strategy uses technical analysis to generate trading signals.
            
            #### Characteristics:
            - Rule-based signal generation
            - Technical indicator driven
            - Deterministic decision making
            - Consistent risk management
            """)
        
        # Performance comparison section
        st.subheader("📈 Performance Breakdown")
        
        # Risk metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Risk-Adjusted Return", f"{results.get('sharpe_ratio', 0):.2f}")
        with col2:
            st.metric("Maximum Drawdown", f"{results.get('max_drawdown', 0):.2%}")
        with col3:
            st.metric("Profit Factor", f"{results.get('profit_factor', 0):.2f}")
        
        # Trading frequency analysis
        if not results['trades'].empty:
            trades_df = results['trades']
            if 'entry_date' in trades_df.columns and 'exit_date' in trades_df.columns:
                try:
                    avg_holding_period = (pd.to_datetime(trades_df['exit_date']) - pd.to_datetime(trades_df['entry_date'])).dt.days.mean()
                    st.info(f"**Average Holding Period:** {avg_holding_period:.1f} days")
                except:
                    st.info("**Average Holding Period:** Not available")
            else:
                st.info("**Trading Activity:** Signal-based entries and exits")
    except Exception as e:
        st.error(f"Error displaying strategy insights: {str(e)}")

if __name__ == "__main__":
    main()
