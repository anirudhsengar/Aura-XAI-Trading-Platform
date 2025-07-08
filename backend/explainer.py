import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import warnings
import shap
from sklearn.ensemble import RandomForestClassifier
from utils import LoggingUtils
from strategies import BaseStrategy

warnings.filterwarnings('ignore')

class Explainer:
    """
    Explainable AI system for trading decision analysis.
    """
    
    def __init__(self, log_level: str = "INFO"):
        """
        Initialize the Explainer.
        
        Args:
            log_level: Logging level
        """
        self.logger = LoggingUtils.setup_logger(
            "Explainer",
            level=log_level
        )
        
        # Initialize SHAP explainer (will be set based on strategy type)
        self.shap_explainer = None
        self.feature_names = []
        self.explanation_model = None
        
        # Explanation cache
        self.explanation_cache = {}
        
        self.logger.info("Explainer initialized successfully")
    
    def explain_trade_decision(self, strategy: BaseStrategy, data: pd.DataFrame,
                              trade_date: datetime, signal: int,
                              context_window: int = 20) -> Dict[str, Any]:
        """
        Explain why a specific trade decision was made.
        
        Args:
            strategy: The trading strategy that generated the signal
            data: Historical data including features
            trade_date: Date of the trade decision
            signal: The trading signal (1 for buy, -1 for sell, 0 for hold)
            context_window: Number of days of context to include
            
        Returns:
            Dict: Comprehensive explanation of the trade decision
        """
        self.logger.info(f"Explaining trade decision for {trade_date}")
        
        # Get the data slice for the trade date
        trade_data = self._get_trade_context(data, trade_date, context_window)
        
        if trade_data.empty:
            self.logger.warning(f"No data available for trade date {trade_date}")
            return {}
        
        # Get the specific row for the trade date
        trade_row = trade_data.loc[trade_date] if trade_date in trade_data.index else trade_data.iloc[-1]
        
        # Generate explanation based on strategy type
        if strategy.name in ['MLStrategy', 'MultiFactor']:
            explanation = self._explain_ml_decision(strategy, trade_data, trade_row, signal)
        else:
            explanation = self._explain_rule_based_decision(strategy, trade_data, trade_row, signal)
        
        # Add common context
        explanation.update({
            'trade_date': trade_date,
            'signal': signal,
            'signal_description': self._get_signal_description(signal),
            'strategy_name': strategy.name,
            'market_context': self._get_market_context(trade_data),
            'risk_factors': self._identify_risk_factors(trade_row)
        })
        
        return explanation
    
    def _explain_ml_decision(self, strategy: BaseStrategy, data: pd.DataFrame,
                            trade_row: pd.Series, signal: int) -> Dict[str, Any]:
        """
        Explain ML-based trading decisions using SHAP.
        """
        try:
            # Prepare features for SHAP analysis
            feature_data = self._prepare_features_for_shap(data)
            
            if feature_data.empty:
                return self._fallback_explanation(trade_row, signal)
            
            # Create a simple model for explanation if strategy doesn't have one
            if not hasattr(strategy, 'models') or not strategy.models:
                explanation_model = self._create_explanation_model(feature_data, data)
            else:
                explanation_model = list(strategy.models.values())[0]  # Use first model
            
            # Get SHAP values
            shap_values = self._calculate_shap_values(explanation_model, feature_data)
            
            # Get the SHAP values for the trade date
            trade_index = feature_data.index.get_loc(trade_row.name) if trade_row.name in feature_data.index else -1
            trade_shap_values = shap_values[trade_index]
            
            # Create feature importance ranking
            feature_importance = self._rank_feature_importance(
                feature_data.columns, trade_shap_values
            )
            
            # Generate explanations
            explanation = {
                'explanation_type': 'ml_based',
                'shap_values': trade_shap_values.tolist(),
                'feature_names': feature_data.columns.tolist(),
                'feature_importance': feature_importance,
                'top_positive_factors': self._get_top_factors(feature_importance, 'positive'),
                'top_negative_factors': self._get_top_factors(feature_importance, 'negative'),
                'decision_summary': self._generate_decision_summary(feature_importance, signal),
                'confidence_score': self._calculate_confidence_score(trade_shap_values)
            }
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error in ML explanation: {str(e)}")
            return self._fallback_explanation(trade_row, signal)
    
    def _explain_rule_based_decision(self, strategy: BaseStrategy, data: pd.DataFrame,
                                   trade_row: pd.Series, signal: int) -> Dict[str, Any]:
        """
        Explain rule-based trading decisions.
        """
        explanation = {
            'explanation_type': 'rule_based',
            'triggered_conditions': [],
            'feature_analysis': {},
            'decision_rationale': []
        }
        
        # Analyze based on strategy type
        if strategy.name == 'MeanReversion':
            explanation.update(self._explain_mean_reversion(trade_row, signal, strategy.params))
        elif strategy.name == 'Momentum':
            explanation.update(self._explain_momentum(trade_row, signal, strategy.params))
        elif strategy.name == 'MultiFactor':
            explanation.update(self._explain_multi_factor(trade_row, signal, strategy.params))
        
        return explanation
    
    def _explain_mean_reversion(self, trade_row: pd.Series, signal: int, params: Dict) -> Dict[str, Any]:
        """Explain mean reversion strategy decisions."""
        explanation = {
            'triggered_conditions': [],
            'feature_analysis': {},
            'decision_rationale': []
        }
        
        # Analyze Bollinger Bands
        if 'BB_Position' in trade_row.index:
            bb_pos = trade_row['BB_Position']
            explanation['feature_analysis']['bollinger_bands'] = {
                'position': bb_pos,
                'interpretation': 'Oversold' if bb_pos < 0.2 else 'Overbought' if bb_pos > 0.8 else 'Normal'
            }
            
            if signal == 1 and bb_pos < 0.2:
                explanation['triggered_conditions'].append('Price below lower Bollinger Band (oversold)')
                explanation['decision_rationale'].append('Mean reversion opportunity: price likely to bounce back')
            elif signal == -1 and bb_pos > 0.8:
                explanation['triggered_conditions'].append('Price above upper Bollinger Band (overbought)')
                explanation['decision_rationale'].append('Mean reversion opportunity: price likely to pull back')
        
        # Analyze RSI
        if 'RSI' in trade_row.index:
            rsi = trade_row['RSI']
            explanation['feature_analysis']['rsi'] = {
                'value': rsi,
                'interpretation': 'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'
            }
            
            if signal == 1 and rsi < params.get('rsi_oversold', 30):
                explanation['triggered_conditions'].append(f'RSI below {params.get("rsi_oversold", 30)} (oversold)')
                explanation['decision_rationale'].append('RSI indicates oversold conditions, potential reversal')
            elif signal == -1 and rsi > params.get('rsi_overbought', 70):
                explanation['triggered_conditions'].append(f'RSI above {params.get("rsi_overbought", 70)} (overbought)')
                explanation['decision_rationale'].append('RSI indicates overbought conditions, potential reversal')
        
        # Analyze Volume
        if 'Volume_Ratio' in trade_row.index:
            vol_ratio = trade_row['Volume_Ratio']
            explanation['feature_analysis']['volume'] = {
                'ratio': vol_ratio,
                'interpretation': 'High' if vol_ratio > 1.5 else 'Normal' if vol_ratio > 0.8 else 'Low'
            }
            
            if vol_ratio > params.get('volume_threshold', 1.5):
                explanation['triggered_conditions'].append('Above-average volume confirms signal')
                explanation['decision_rationale'].append('High volume increases confidence in reversal')
        
        return explanation
    
    def _explain_momentum(self, trade_row: pd.Series, signal: int, params: Dict) -> Dict[str, Any]:
        """Explain momentum strategy decisions."""
        explanation = {
            'triggered_conditions': [],
            'feature_analysis': {},
            'decision_rationale': []
        }
        
        # Analyze Moving Averages
        if 'SMA_10' in trade_row.index and 'SMA_50' in trade_row.index:
            sma_10 = trade_row['SMA_10']
            sma_50 = trade_row['SMA_50']
            price = trade_row['Close']
            
            explanation['feature_analysis']['moving_averages'] = {
                'price_vs_sma10': 'Above' if price > sma_10 else 'Below',
                'price_vs_sma50': 'Above' if price > sma_50 else 'Below',
                'sma10_vs_sma50': 'Above' if sma_10 > sma_50 else 'Below'
            }
            
            if signal == 1:
                if price > sma_10 > sma_50:
                    explanation['triggered_conditions'].append('Price above both moving averages (bullish alignment)')
                    explanation['decision_rationale'].append('Strong upward momentum confirmed by MA alignment')
            elif signal == -1:
                if price < sma_10 < sma_50:
                    explanation['triggered_conditions'].append('Price below both moving averages (bearish alignment)')
                    explanation['decision_rationale'].append('Strong downward momentum confirmed by MA alignment')
        
        # Analyze MACD
        if 'MACD' in trade_row.index and 'MACD_Signal' in trade_row.index:
            macd = trade_row['MACD']
            macd_signal = trade_row['MACD_Signal']
            
            explanation['feature_analysis']['macd'] = {
                'macd_value': macd,
                'signal_value': macd_signal,
                'relationship': 'Above' if macd > macd_signal else 'Below'
            }
            
            if signal == 1 and macd > macd_signal:
                explanation['triggered_conditions'].append('MACD above signal line (bullish)')
                explanation['decision_rationale'].append('MACD confirms upward momentum')
            elif signal == -1 and macd < macd_signal:
                explanation['triggered_conditions'].append('MACD below signal line (bearish)')
                explanation['decision_rationale'].append('MACD confirms downward momentum')
        
        return explanation
    
    def _explain_multi_factor(self, trade_row: pd.Series, signal: int, params: Dict) -> Dict[str, Any]:
        """Explain multi-factor strategy decisions."""
        explanation = {
            'triggered_conditions': [],
            'feature_analysis': {},
            'decision_rationale': [],
            'factor_scores': {}
        }
        
        # Technical factor analysis
        technical_score = 0
        if 'RSI' in trade_row.index:
            rsi_score = (trade_row['RSI'] - 50) / 50
            technical_score += rsi_score * 0.3
            explanation['factor_scores']['rsi_contribution'] = rsi_score * 0.3
        
        # Sentiment factor analysis
        sentiment_score = 0
        if 'sentiment_mean' in trade_row.index:
            sentiment_score = trade_row['sentiment_mean']
            explanation['factor_scores']['sentiment_contribution'] = sentiment_score * params.get('sentiment_weight', 0.3)
        
        # Combine factor explanations
        explanation['feature_analysis']['multi_factor'] = {
            'technical_score': technical_score,
            'sentiment_score': sentiment_score,
            'combined_signal_strength': abs(technical_score) + abs(sentiment_score)
        }
        
        if signal != 0:
            explanation['decision_rationale'].append(f'Multi-factor model combines technical and sentiment signals')
            explanation['triggered_conditions'].append(f'Composite score exceeded threshold')
        
        return explanation
    
    def _calculate_shap_values(self, model, feature_data: pd.DataFrame) -> np.ndarray:
        """Calculate SHAP values for the model."""
        try:
            # Create SHAP explainer
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.LinearExplainer(model, feature_data)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(feature_data)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class
            
            return shap_values
            
        except Exception as e:
            self.logger.error(f"Error calculating SHAP values: {str(e)}")
            # Return zeros if SHAP calculation fails
            return np.zeros((len(feature_data), len(feature_data.columns)))
    
    def _create_explanation_model(self, feature_data: pd.DataFrame, 
                                 original_data: pd.DataFrame) -> Any:
        """Create a simple model for explanation purposes."""
        try:
            # Create labels based on price movements
            future_returns = original_data['Close'].shift(-1) / original_data['Close'] - 1
            labels = np.where(future_returns > 0.01, 1, 0)
            
            # Align with feature data
            common_idx = feature_data.index.intersection(original_data.index)
            X = feature_data.loc[common_idx]
            y = labels[:len(X)]
            
            # Train simple model
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating explanation model: {str(e)}")
            return None
    
    def _prepare_features_for_shap(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature data for SHAP analysis."""
        # Select relevant features
        feature_columns = [
            'RSI', 'MACD', 'BB_Position', 'Volume_Ratio', 'Price_Change',
            'sentiment_mean', 'SMA_10', 'SMA_50', 'Volatility_20'
        ]
        
        available_features = [col for col in feature_columns if col in data.columns]
        
        if not available_features:
            return pd.DataFrame()
        
        feature_data = data[available_features].copy()
        
        # Fill NaN values
        feature_data = feature_data.fillna(method='ffill').fillna(0)
        
        return feature_data
    
    def _rank_feature_importance(self, feature_names: List[str], 
                                shap_values: np.ndarray) -> List[Dict[str, Any]]:
        """Rank features by their SHAP value importance."""
        importance_data = []
        
        for i, feature in enumerate(feature_names):
            shap_value = shap_values[i]
            importance_data.append({
                'feature': feature,
                'shap_value': float(shap_value),
                'abs_importance': abs(shap_value),
                'contribution': 'positive' if shap_value > 0 else 'negative',
                'description': self._get_feature_description(feature)
            })
        
        # Sort by absolute importance
        importance_data.sort(key=lambda x: x['abs_importance'], reverse=True)
        
        return importance_data
    
    def _get_feature_description(self, feature: str) -> str:
        """Get human-readable description of features."""
        descriptions = {
            'RSI': 'Relative Strength Index - measures overbought/oversold conditions',
            'MACD': 'Moving Average Convergence Divergence - trend and momentum indicator',
            'BB_Position': 'Bollinger Bands Position - price position within bands',
            'Volume_Ratio': 'Volume Ratio - current volume vs. average volume',
            'Price_Change': 'Price Change - recent price movement',
            'sentiment_mean': 'News Sentiment - average sentiment from news analysis',
            'SMA_10': '10-day Simple Moving Average',
            'SMA_50': '50-day Simple Moving Average',
            'Volatility_20': '20-day Price Volatility'
        }
        
        return descriptions.get(feature, f'{feature} - technical indicator')
    
    def _get_top_factors(self, feature_importance: List[Dict], factor_type: str, 
                        top_n: int = 3) -> List[Dict[str, Any]]:
        """Get top positive or negative contributing factors."""
        filtered_factors = [
            f for f in feature_importance 
            if f['contribution'] == factor_type
        ]
        
        return filtered_factors[:top_n]
    
    def _generate_decision_summary(self, feature_importance: List[Dict], signal: int) -> str:
        """Generate a human-readable summary of the decision."""
        if signal == 0:
            return "No clear signal generated - factors were balanced"
        
        signal_type = "BUY" if signal == 1 else "SELL"
        top_factor = feature_importance[0] if feature_importance else None
        
        if top_factor:
            return f"{signal_type} signal primarily driven by {top_factor['feature']} " \
                   f"({top_factor['description']})"
        else:
            return f"{signal_type} signal generated based on combined factors"
    
    def _calculate_confidence_score(self, shap_values: np.ndarray) -> float:
        """Calculate confidence score based on SHAP values."""
        total_contribution = np.sum(np.abs(shap_values))
        max_contribution = np.max(np.abs(shap_values))
        
        # Confidence is higher when one factor dominates
        confidence = max_contribution / total_contribution if total_contribution > 0 else 0
        
        return min(confidence, 1.0)
    
    def _get_signal_description(self, signal: int) -> str:
        """Get human-readable signal description."""
        signal_map = {
            1: "BUY - Strategy recommends buying the asset",
            -1: "SELL - Strategy recommends selling the asset",
            0: "HOLD - No clear signal, maintain current position"
        }
        
        return signal_map.get(signal, "Unknown signal")
    
    def _get_market_context(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get market context for the trade period."""
        context = {}
        
        if 'Close' in data.columns:
            recent_return = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
            context['recent_performance'] = f"{recent_return:.2%}"
            context['trend'] = "Upward" if recent_return > 0.02 else "Downward" if recent_return < -0.02 else "Sideways"
        
        if 'Volume_Ratio' in data.columns:
            avg_volume = data['Volume_Ratio'].mean()
            context['volume_environment'] = "High" if avg_volume > 1.2 else "Normal" if avg_volume > 0.8 else "Low"
        
        if 'Volatility_20' in data.columns:
            volatility = data['Volatility_20'].iloc[-1]
            context['volatility_level'] = "High" if volatility > 0.03 else "Normal" if volatility > 0.01 else "Low"
        
        return context
    
    def _identify_risk_factors(self, trade_row: pd.Series) -> List[str]:
        """Identify potential risk factors for the trade."""
        risk_factors = []
        
        # High volatility risk
        if 'Volatility_20' in trade_row.index and trade_row['Volatility_20'] > 0.04:
            risk_factors.append("High volatility increases trade risk")
        
        # Low volume risk
        if 'Volume_Ratio' in trade_row.index and trade_row['Volume_Ratio'] < 0.5:
            risk_factors.append("Low volume may impact trade execution")
        
        # Extreme RSI risk
        if 'RSI' in trade_row.index:
            if trade_row['RSI'] > 80:
                risk_factors.append("Extremely overbought conditions")
            elif trade_row['RSI'] < 20:
                risk_factors.append("Extremely oversold conditions")
        
        # Sentiment risk
        if 'sentiment_mean' in trade_row.index:
            sentiment = trade_row['sentiment_mean']
            if abs(sentiment) > 0.5:
                risk_factors.append("Extreme sentiment may indicate market overreaction")
        
        return risk_factors
    
    def _fallback_explanation(self, trade_row: pd.Series, signal: int) -> Dict[str, Any]:
        """Provide fallback explanation when advanced analysis fails."""
        return {
            'explanation_type': 'basic',
            'signal': signal,
            'signal_description': self._get_signal_description(signal),
            'available_features': list(trade_row.index),
            'message': 'Basic explanation - advanced analysis not available'
        }
    
    def _get_trade_context(self, data: pd.DataFrame, trade_date: datetime, 
                          context_window: int) -> pd.DataFrame:
        """Get data context around the trade date."""
        try:
            # Find the closest date in the data
            if trade_date in data.index:
                trade_idx = data.index.get_loc(trade_date)
            else:
                # Find closest date
                closest_idx = data.index.get_indexer([trade_date], method='nearest')[0]
                trade_idx = closest_idx
            
            # Get context window
            start_idx = max(0, trade_idx - context_window)
            end_idx = min(len(data), trade_idx + 1)
            
            return data.iloc[start_idx:end_idx]
            
        except Exception as e:
            self.logger.error(f"Error getting trade context: {str(e)}")
            return pd.DataFrame()