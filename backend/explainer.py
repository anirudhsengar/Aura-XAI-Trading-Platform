import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

class TradingStrategyExplainer:
    """
    Explainer for AI trading strategies providing interpretable insights.
    """
    
    def __init__(self, strategy, data: pd.DataFrame):
        """
        Initialize the explainer.
        
        Args:
            strategy: Strategy instance with explanation capabilities
            data: Market data used for analysis
        """
        self.strategy = strategy
        self.data = data
        self.strategy_name = strategy.name if hasattr(strategy, 'name') else "Unknown"
    
    def generate_comprehensive_explanation(self, date: datetime = None) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a specific prediction.
        """
        try:
            explanation = {
                'strategy_name': self.strategy_name,
                'analysis_date': date or datetime.now(),
                'narrative': '',
                'feature_importance': {},
                'market_context': {},
                'confidence_metrics': {},
                'visualizations': {}
            }
            
            # Get feature importance
            if hasattr(self.strategy, 'get_feature_importance'):
                explanation['feature_importance'] = self.strategy.get_feature_importance(date)
            
            # Generate narrative explanation
            explanation['narrative'] = self._generate_narrative(explanation['feature_importance'], date)
            
            # Get market context
            explanation['market_context'] = self._get_market_context(date)
            
            # Calculate confidence metrics
            explanation['confidence_metrics'] = self._calculate_confidence_metrics(date)
            
            # Generate visualizations
            explanation['visualizations'] = self._generate_visualizations(explanation['feature_importance'])
            
            return explanation
            
        except Exception as e:
            return {'error': f"Failed to generate explanation: {str(e)}"}
    
    def _generate_narrative(self, feature_importance: Dict[str, float], date: datetime = None) -> str:
        """
        Generate human-readable narrative explanation.
        """
        if not feature_importance:
            # Check if we have any explanation data from the strategy
            if hasattr(self.strategy, 'explanation_cache') and self.strategy.explanation_cache:
                return self._generate_lstm_narrative(date)
            return "No feature importance data available for explanation."
        
        # Get top features
        top_features = list(feature_importance.items())[:5]
        
        narrative = f"## AI Model Decision Analysis\n\n"
        narrative += f"**Strategy:** {self.strategy_name}\n"
        if date:
            narrative += f"**Analysis Date:** {date.strftime('%Y-%m-%d')}\n\n"
        
        narrative += "### Key Factors Influencing the Decision:\n\n"
        
        for i, (feature, importance) in enumerate(top_features, 1):
            feature_desc = self._describe_feature(feature)
            impact_level = self._get_impact_level(importance)
            narrative += f"{i}. **{feature_desc}** - {impact_level} impact ({importance:.1%})\n"
        
        # Add interpretation based on strategy type
        if 'lstm' in self.strategy_name.lower():
            narrative += self._generate_lstm_narrative(date)
        elif 'mean' in self.strategy_name.lower():
            narrative += "\n### Mean Reversion Strategy Insights:\n"
            narrative += "The strategy identified oversold/overbought conditions suggesting a price reversion "
            narrative += "opportunity based on statistical analysis of price movements relative to historical norms.\n"
        elif 'momentum' in self.strategy_name.lower():
            narrative += "\n### Momentum Strategy Insights:\n"
            narrative += "The strategy detected strong directional momentum in multiple indicators, "
            narrative += "suggesting the current trend is likely to continue in the near term.\n"
        elif 'ma' in self.strategy_name.lower():
            narrative += "\n### Moving Average Strategy Insights:\n"
            narrative += "The strategy identified a significant crossover in moving averages, "
            narrative += "indicating a potential change in trend direction.\n"
        
        return narrative

    def _generate_lstm_narrative(self, date: datetime = None) -> str:
        """
        Generate specialized narrative for LSTM strategy explanations.
        """
        if not hasattr(self.strategy, 'explanation_cache') or not self.strategy.explanation_cache:
            return "\n### LSTM Model Insights:\nNo detailed explanations available."
        
        try:
            # Get the relevant explanation
            explanation = None
            if date and date in self.strategy.explanation_cache:
                explanation = self.strategy.explanation_cache[date]
            else:
                # Use most recent explanation
                latest_date = max(self.strategy.explanation_cache.keys())
                explanation = self.strategy.explanation_cache[latest_date]
            
            if not explanation:
                return "\n### LSTM Model Insights:\nNo explanation data found."
            
            narrative = "\n### LSTM Neural Network Analysis:\n\n"
            
            # Add prediction details
            predicted_class = explanation.get('predicted_class', 0)
            max_probability = explanation.get('max_probability', 0.0)
            signal_type = explanation.get('signal_type', 'trade')
            
            signal_names = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
            signal_name = signal_names.get(predicted_class, 'UNKNOWN')
            
            narrative += f"**Prediction:** {signal_name} signal with {max_probability:.1%} confidence\n"
            narrative += f"**Signal Type:** {signal_type.replace('_', ' ').title()}\n\n"
            
            # Confidence assessment
            if max_probability > 0.8:
                confidence_desc = "very high confidence"
            elif max_probability > 0.6:
                confidence_desc = "moderate confidence"
            else:
                confidence_desc = "low confidence"
            
            narrative += f"The LSTM neural network analyzed {self.strategy.params.get('lookback_window', 60)} "
            narrative += f"time steps of market data and made this prediction with {confidence_desc}. "
            
            # Add feature analysis if available
            features = explanation.get('features', [])
            if features:
                narrative += f"The model considered {len(features)} technical indicators including "
                narrative += "price movements, volume patterns, and momentum indicators to identify "
                narrative += "complex sequential patterns that traditional methods might miss.\n\n"
            
            # Add interpretation based on predicted class
            if predicted_class == 1:
                narrative += "**Bullish Pattern Detected:** The model identified patterns suggesting upward price momentum. "
                narrative += "Multiple indicators align to suggest buying pressure may increase.\n"
            elif predicted_class == -1:
                narrative += "**Bearish Pattern Detected:** The model identified patterns suggesting downward price pressure. "
                narrative += "Multiple indicators align to suggest selling pressure may increase.\n"
            else:
                narrative += "**Neutral Pattern:** The model suggests the current market conditions don't strongly "
                narrative += "favor either direction, indicating a consolidation or uncertain phase.\n"
            
            # Add timing context
            prediction_date = explanation.get('prediction_date')
            if prediction_date:
                narrative += f"\n**Analysis Timestamp:** {prediction_date}\n"
            
            # Add model performance context
            narrative += "\n**Model Context:** This LSTM model was trained on historical price patterns and "
            narrative += "technical indicators. It uses deep learning to capture non-linear relationships "
            narrative += "and temporal dependencies that may not be apparent to traditional analysis methods.\n"
            
            return narrative
            
        except Exception as e:
            return f"\n### LSTM Model Insights:\nError generating narrative: {str(e)}\n"

    def _describe_feature(self, feature: str) -> str:
        """
        Convert technical feature names to human-readable descriptions.
        """
        feature_descriptions = {
            'rsi': 'Relative Strength Index (RSI)',
            'macd': 'MACD Momentum Indicator',
            'bb_position': 'Bollinger Bands Position',
            'volume_ratio': 'Volume Relative to Average',
            'volatility': 'Price Volatility',
            'price_momentum': 'Price Momentum',
            'ma_ratio': 'Moving Average Ratio',
            'atr': 'Average True Range',
            'return_1d': '1-Day Price Return',
            'return_5d': '5-Day Price Return',
            'return_20d': '20-Day Price Return',
            'stoch_k': 'Stochastic Oscillator',
            'williams_r': 'Williams %R',
            'garman_klass_vol': 'Garman-Klass Volatility'
        }
        
        # Try exact match first
        if feature in feature_descriptions:
            return feature_descriptions[feature]
        
        # Try partial matches
        for key, desc in feature_descriptions.items():
            if key in feature.lower():
                return desc
        
        # Default: clean up the feature name
        return feature.replace('_', ' ').title()
    
    def _get_impact_level(self, importance: float) -> str:
        """
        Convert importance score to descriptive impact level.
        """
        if importance > 0.3:
            return "Very high"
        elif importance > 0.2:
            return "High"
        elif importance > 0.1:
            return "Medium"
        elif importance > 0.05:
            return "Low"
        else:
            return "Very low"
    
    def _get_market_context(self, date: datetime = None) -> Dict[str, Any]:
        """
        Get current market context for explanation.
        """
        try:
            # Use the most recent data if no date specified
            if date is None or date not in self.data.index:
                current_data = self.data.iloc[-1]
                analysis_date = self.data.index[-1]
            else:
                current_data = self.data.loc[date]
                analysis_date = date
            
            context = {
                'current_price': float(current_data['Close']),
                'analysis_date': analysis_date
            }
            
            # Calculate trend over different periods
            if len(self.data) >= 30:
                price_30d_ago = self.data['Close'].iloc[-30] if date is None else \
                               self.data.loc[:date]['Close'].iloc[-30] if len(self.data.loc[:date]) >= 30 else \
                               self.data['Close'].iloc[0]
                context['trend_30d'] = (current_data['Close'] - price_30d_ago) / price_30d_ago
            
            # Volatility metrics
            if 'Volatility_20' in self.data.columns:
                context['volatility_annualized'] = float(current_data.get('Volatility_20', 0)) * np.sqrt(252)
            else:
                returns = self.data['Close'].pct_change().dropna()
                if len(returns) > 20:
                    context['volatility_annualized'] = float(returns.rolling(20).std().iloc[-1] * np.sqrt(252))
            
            # Price position relative to recent range
            if len(self.data) >= 50:
                recent_data = self.data.tail(50)
                recent_high = recent_data['Close'].max()
                recent_low = recent_data['Close'].min()
                if recent_high != recent_low:
                    context['price_position'] = (current_data['Close'] - recent_low) / (recent_high - recent_low)
                else:
                    context['price_position'] = 0.5
            
            return context
            
        except Exception as e:
            return {'error': f"Failed to get market context: {str(e)}"}
    
    def _calculate_confidence_metrics(self, date: datetime = None) -> Dict[str, float]:
        """
        Calculate confidence metrics for the prediction.
        """
        try:
            metrics = {}
            
            # Enhanced LSTM confidence metrics
            if hasattr(self.strategy, 'explanation_cache') and self.strategy.explanation_cache:
                explanation = None
                if date and date in self.strategy.explanation_cache:
                    explanation = self.strategy.explanation_cache[date]
                elif self.strategy.explanation_cache:
                    # Use most recent explanation
                    latest_date = max(self.strategy.explanation_cache.keys())
                    explanation = self.strategy.explanation_cache[latest_date]
                
                if explanation:
                    # Direct prediction confidence
                    max_prob = explanation.get('max_probability', 0.0)
                    if max_prob > 0:
                        metrics['prediction_confidence'] = max_prob
                    
                    # SHAP-based confidence
                    shap_values = explanation.get('shap_values', [])
                    if self._has_values(shap_values):
                        try:
                            # Normalize to iterable of arrays
                            iterable = shap_values if isinstance(shap_values, (list, tuple)) else [shap_values]
                            total_shap = 0.0
                            count = 0
                            for class_shap in iterable:
                                if class_shap is not None:
                                    total_shap += float(np.mean(np.abs(class_shap)))
                                    count += 1
                            if count > 0:
                                avg_shap_magnitude = total_shap / count
                                metrics['shap_confidence'] = min(avg_shap_magnitude * 2, 1.0)
                        except Exception as e:
                            print(f"SHAP confidence calculation error: {str(e)}")
                            metrics['shap_confidence'] = 0.5
                    
                    # Feature consistency confidence
                    features = explanation.get('features', [])
                    if features and shap_values:
                        try:
                            # Calculate how concentrated the importance is in top features
                            feature_importance = self.strategy.get_feature_importance(date)
                            if feature_importance:
                                total_importance = sum(feature_importance.values())
                                if total_importance > 0:
                                    top_5_importance = sum(list(feature_importance.values())[:5])
                                    metrics['feature_consistency'] = top_5_importance / total_importance
                        except Exception:
                            metrics['feature_consistency'] = 0.5
            
            # Strategy-specific confidence metrics
            if hasattr(self.strategy, 'current_position') and self.strategy.current_position != 0:
                # Position performance confidence
                if hasattr(self.strategy, 'entry_price') and self.strategy.entry_price:
                    try:
                        current_price = float(self.data['Close'].iloc[-1])
                        pnl_pct = (current_price - self.strategy.entry_price) / self.strategy.entry_price
                        # Normalize PnL to confidence score
                        metrics['position_confidence'] = max(0.1, min(1.0, 0.5 + pnl_pct * 2))
                    except Exception:
                        metrics['position_confidence'] = 0.5
            
            # Overall confidence calculation
            if metrics:
                weights = {
                    'prediction_confidence': 0.4,
                    'shap_confidence': 0.3,
                    'feature_consistency': 0.2,
                    'position_confidence': 0.1
                }
                
                weighted_sum = 0
                total_weight = 0
                
                for metric, value in metrics.items():
                    weight = weights.get(metric, 0.1)
                    weighted_sum += value * weight
                    total_weight += weight
                
                if total_weight > 0:
                    metrics['overall_confidence'] = weighted_sum / total_weight
                else:
                    metrics['overall_confidence'] = np.mean(list(metrics.values()))
            else:
                metrics['overall_confidence'] = 0.5  # Default neutral confidence
            
            return metrics
            
        except Exception as e:
            print(f"Confidence metrics calculation error: {str(e)}")
            return {'overall_confidence': 0.5}
    
    def _generate_visualizations(self, feature_importance: Dict[str, float]) -> Dict[str, str]:
        """
        Generate visualization HTML for explanations.
        """
        visualizations = {}
        
        try:
            if feature_importance and len(feature_importance) > 0:
                # Feature importance waterfall chart
                top_features = dict(list(feature_importance.items())[:10])
                
                if top_features:
                    fig = go.Figure(go.Waterfall(
                        name="Feature Impact",
                        orientation="v",
                        measure=["relative"] * len(top_features),
                        x=list(top_features.keys()),
                        textposition="outside",
                        text=[f"{v:.1%}" for v in top_features.values()],
                        y=list(top_features.values()),
                        connector={"line": {"color": "rgb(63, 63, 63)"}},
                    ))
                    
                    fig.update_layout(
                        title="Feature Importance Breakdown",
                        showlegend=False,
                        xaxis_tickangle=-45,
                        height=500
                    )
                    
                    try:
                        from plotly.offline import plot
                        visualizations['feature_importance'] = plot(fig, output_type='div', include_plotlyjs=False)
                    except ImportError:
                        visualizations['feature_importance'] = "<div>Plotly visualization not available</div>"
            
            return visualizations
            
        except Exception as e:
            print(f"Visualization generation error: {str(e)}")
            return {'error': f"Failed to generate visualizations: {str(e)}"}
    
    def export_explanation_report(self, date: Optional[datetime] = None, format: str = 'html') -> str:
        """
        Export comprehensive explanation report.
        """
        try:
            explanation = self.generate_comprehensive_explanation(date)
            
            if "error" in explanation:
                if format.lower() == 'html':
                    return f"<html><body><h1>Error</h1><p>{explanation['error']}</p></body></html>"
                else:
                    return f"Error: {explanation['error']}"
            
            if format.lower() == 'html':
                return self._generate_html_report(explanation)
            elif format.lower() == 'json':
                import json
                return json.dumps(explanation, default=str, indent=2)
            else:
                return str(explanation)
                
        except Exception as e:
            error_msg = f"Report generation failed: {str(e)}"
            if format.lower() == 'html':
                return f"<html><body><h1>Report Generation Failed</h1><p>{error_msg}</p></body></html>"
            else:
                return error_msg
    
    def _generate_html_report(self, explanation: Dict) -> str:
        """
        Generate HTML report from explanation data.
        """
        try:
            html_parts = [
                "<html><head><title>Trading Strategy Explanation Report</title>",
                "<style>",
                "body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }",
                "h1, h2, h3 { color: #333; }",
                ".section { margin: 20px 0; padding: 15px; border-left: 4px solid #007bff; background: #f8f9fa; }",
                ".metric { background: #ffffff; padding: 10px; margin: 5px 0; border-radius: 5px; border: 1px solid #dee2e6; }",
                ".narrative { background: #ffffff; padding: 15px; border-radius: 5px; }",
                "strong { color: #495057; }",
                ".error { color: #dc3545; background: #f8d7da; padding: 10px; border-radius: 5px; }",
                "</style></head><body>",
                
                f"<h1>Trading Strategy Explanation Report</h1>",
                f"<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
                f"<p><strong>Strategy:</strong> {explanation.get('strategy_name', 'Unknown')}</p>"
            ]
            
            # Add narrative with proper markdown conversion
            if 'narrative' in explanation and explanation['narrative']:
                html_parts.append('<div class="section">')
                html_parts.append('<h2>Analysis Summary</h2>')
                html_parts.append('<div class="narrative">')
                narrative_html = self._convert_markdown_to_html(explanation['narrative'])
                html_parts.append(narrative_html)
                html_parts.append('</div>')
                html_parts.append('</div>')
            
            # Add market context
            if 'market_context' in explanation and explanation['market_context']:
                context = explanation['market_context']
                if 'error' not in context:
                    html_parts.append('<div class="section">')
                    html_parts.append('<h2>Market Context</h2>')
                    for key, value in context.items():
                        if isinstance(value, float):
                            if 'price' in key.lower():
                                value_str = f"${value:.2f}"
                            elif 'trend' in key.lower() or 'position' in key.lower():
                                value_str = f"{value:.2%}"
                            else:
                                value_str = f"{value:.4f}"
                        else:
                            value_str = str(value)
                        html_parts.append(f'<div class="metric"><strong>{key.replace("_", " ").title()}:</strong> {value_str}</div>')
                    html_parts.append('</div>')
            
            # Add confidence metrics
            if 'confidence_metrics' in explanation and explanation['confidence_metrics']:
                html_parts.append('<div class="section">')
                html_parts.append('<h2>Confidence Metrics</h2>')
                for key, value in explanation['confidence_metrics'].items():
                    if isinstance(value, (int, float)):
                        html_parts.append(f'<div class="metric"><strong>{key.replace("_", " ").title()}:</strong> {value:.1%}</div>')
                html_parts.append('</div>')
            
            # Add visualizations
            if 'visualizations' in explanation and explanation['visualizations']:
                html_parts.append('<div class="section">')
                html_parts.append('<h2>Visualizations</h2>')
                for viz_name, viz_html in explanation['visualizations'].items():
                    if viz_html and 'error' not in viz_html.lower():
                        html_parts.append(f'<h3>{viz_name.replace("_", " ").title()}</h3>')
                        html_parts.append(viz_html)
                    elif viz_html:
                        html_parts.append(f'<div class="error">{viz_html}</div>')
                html_parts.append('</div>')
            
            html_parts.append("</body></html>")
            
            return "\n".join(html_parts)
            
        except Exception as e:
            return f"<html><body><h1>Error generating report</h1><p>{str(e)}</p></body></html>"
    
    def _convert_markdown_to_html(self, text: str) -> str:
        """
        Convert basic markdown to HTML.
        """
        # Convert markdown headers
        text = text.replace('### ', '<h3>').replace('\n', '</h3>\n', text.count('### '))
        text = text.replace('## ', '<h2>').replace('\n', '</h2>\n', text.count('## '))
        
        # Convert bold text
        import re
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        
        # Convert line breaks
        text = text.replace('\n', '<br>')
        
        # Fix header closing tags
        text = re.sub(r'<h([23])>(.*?)</h\1><br>', r'<h\1>\2</h\1>', text)
        
        return text

    def _has_values(self, v) -> bool:
        """
        Safely check if a value/list/ndarray has content.
        """
        try:
            if v is None:
                return False
            if isinstance(v, (list, tuple)):
                return len(v) > 0
            if isinstance(v, np.ndarray):
                return v.size > 0
            return bool(v)
        except Exception:
            return False

class SHAPAnalyzer:
    """
    Specialized SHAP analyzer for trading strategies.
    """
    
    def __init__(self, strategy):
        self.strategy = strategy
        
    def analyze_feature_interactions(self, feature_data: pd.DataFrame, 
                                   max_interactions: int = 10) -> Dict[str, float]:
        """
        Analyze feature interactions using SHAP.
        """
        try:
            if not hasattr(self.strategy, 'shap_explainer') or self.strategy.shap_explainer is None:
                print("SHAP explainer not available for interaction analysis")
                return {}
            
            # Check if we have SHAP available
            try:
                import shap
            except ImportError:
                print("SHAP library not available")
                return {}
            
            # Get recent explanations for interaction analysis
            if hasattr(self.strategy, 'explanation_cache') and self.strategy.explanation_cache:
                interactions = {}
                
                # Analyze interactions from cached explanations
                for date, explanation in list(self.strategy.explanation_cache.items())[-5:]:  # Last 5 explanations
                    shap_values = explanation.get('shap_values', [])
                    features = explanation.get('features', [])
                    
                    if shap_values and features:
                        try:
                            # Calculate feature interaction strength
                            for i, feature1 in enumerate(features[:max_interactions]):
                                for j, feature2 in enumerate(features[i+1:max_interactions], i+1):
                                    interaction_key = f"{feature1}_x_{feature2}"
                                    
                                    # Calculate interaction strength across all classes
                                    interaction_strength = 0
                                    for class_shap in shap_values:
                                        if (class_shap is not None and 
                                            i < class_shap.shape[1] and 
                                            j < class_shap.shape[1]):
                                            # Simple interaction measure: correlation of SHAP values
                                            corr = np.corrcoef(class_shap[:, i], class_shap[:, j])[0, 1]
                                            if not np.isnan(corr):
                                                interaction_strength += abs(corr)
                                    
                                    if interaction_key not in interactions:
                                        interactions[interaction_key] = []
                                    interactions[interaction_key].append(interaction_strength)
                        
                        except Exception as e:
                            print(f"Error analyzing interactions for {date}: {str(e)}")
                            continue
                
                # Average interaction strengths across time
                avg_interactions = {}
                for key, values in interactions.items():
                    if values:
                        avg_interactions[key] = np.mean(values)
                
                # Return top interactions
                sorted_interactions = dict(sorted(avg_interactions.items(), 
                                                key=lambda x: x[1], reverse=True))
                return dict(list(sorted_interactions.items())[:max_interactions])
            
            return {}
            
        except Exception as e:
            print(f"Feature interaction analysis error: {str(e)}")
            return {}
    
    def generate_shap_dashboard(self) -> str:
        """
        Generate interactive SHAP dashboard with comprehensive visualizations.
        """
        try:
            dashboard_parts = [
                '<div id="shap-dashboard" style="padding: 20px; font-family: Arial, sans-serif;">',
                '<h2 style="color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px;">SHAP Analysis Dashboard</h2>'
            ]
            
            # Check if SHAP explainer is available
            if not hasattr(self.strategy, 'shap_explainer') or self.strategy.shap_explainer is None:
                dashboard_parts.extend([
                    '<div style="background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 10px 0;">',
                    '<strong>Notice:</strong> SHAP explainer not available for this strategy.',
                    '</div>',
                    '</div>'
                ])
                return '\n'.join(dashboard_parts)
            
            # Feature importance section
            if hasattr(self.strategy, 'get_feature_importance'):
                try:
                    feature_importance = self.strategy.get_feature_importance()
                    if feature_importance:
                        dashboard_parts.extend([
                            '<div style="margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 5px;">',
                            '<h3 style="color: #495057;">Feature Importance Summary</h3>',
                            '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">'
                        ])
                        
                        for feature, importance in list(feature_importance.items())[:8]:
                            dashboard_parts.append(
                                f'<div style="background: white; padding: 10px; border-radius: 3px; border-left: 4px solid #007bff;">'
                                f'<strong>{feature.replace("_", " ").title()}</strong><br>'
                                f'<span style="font-size: 1.2em; color: #28a745;">{importance:.1%}</span>'
                                f'</div>'
                            )
                        
                        dashboard_parts.extend(['</div>', '</div>'])
                except Exception as e:
                    print(f"Error generating feature importance section: {str(e)}")
            
            # SHAP explanations section
            if hasattr(self.strategy, 'explanation_cache') and self.strategy.explanation_cache:
                dashboard_parts.extend([
                    '<div style="margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 5px;">',
                    '<h3 style="color: #495057;">Recent SHAP Explanations</h3>'
                ])
                
                recent_explanations = list(self.strategy.explanation_cache.items())[-3:]  # Last 3
                for date, explanation in recent_explanations:
                    shap_values = explanation.get('shap_values', [])
                    if self._has_values(shap_values):
                        dashboard_parts.extend([
                            f'<div style="background: white; margin: 10px 0; padding: 10px; border-radius: 3px;">',
                            f'<h4 style="margin: 0 0 10px 0; color: #6c757d;">Prediction: {date}</h4>',
                            f'<p style="margin: 5px 0;">Classes analyzed: {len(shap_values)}</p>',
                            f'<p style="margin: 5px 0;">Features: {len(explanation.get("features", []))}</p>',
                            '</div>'
                        ])
                
                dashboard_parts.append('</div>')
            
            # Feature interactions section
            try:
                interactions = self.analyze_feature_interactions(pd.DataFrame(), max_interactions=5)
                if interactions:
                    dashboard_parts.extend([
                        '<div style="margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 5px;">',
                        '<h3 style="color: #495057;">Feature Interactions</h3>',
                        '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px;">'
                    ])
                    
                    for interaction, strength in interactions.items():
                        features = interaction.split('_x_')
                        if len(features) == 2:
                            dashboard_parts.append(
                                f'<div style="background: white; padding: 10px; border-radius: 3px; border-left: 4px solid #ffc107;">'
                                f'<strong>{features[0]} ↔ {features[1]}</strong><br>'
                                f'<span style="font-size: 1.1em; color: #fd7e14;">Strength: {strength:.3f}</span>'
                                f'</div>'
                            )
                    
                    dashboard_parts.extend(['</div>', '</div>'])
            except Exception as e:
                print(f"Error generating interactions section: {str(e)}")
            
            # Model confidence section
            try:
                if hasattr(self.strategy, 'explanation_cache') and self.strategy.explanation_cache:
                    latest_date = max(self.strategy.explanation_cache.keys())
                    latest_explanation = self.strategy.explanation_cache[latest_date]
                    
                    dashboard_parts.extend([
                        '<div style="margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 5px;">',
                        '<h3 style="color: #495057;">Model Confidence Metrics</h3>',
                        f'<p><strong>Latest Analysis:</strong> {latest_date}</p>'
                    ])
                    
                    shap_values = latest_explanation.get('shap_values', [])
                    if self._has_values(shap_values):
                        try:
                            # Normalize iterable
                            iterable = shap_values if isinstance(shap_values, (list, tuple)) else [shap_values]
                            total_magnitude = 0.0
                            for class_shap in iterable:
                                if class_shap is not None:
                                    total_magnitude += float(np.mean(np.abs(class_shap)))
                            confidence_level = min(total_magnitude * 100, 100)
                            confidence_color = "#28a745" if confidence_level > 70 else "#ffc107" if confidence_level > 40 else "#dc3545"
                            
                            dashboard_parts.extend([
                                f'<div style="background: white; padding: 15px; border-radius: 3px; text-align: center;">',
                                f'<div style="font-size: 2em; color: {confidence_color}; font-weight: bold;">{confidence_level:.1f}%</div>',
                                f'<div style="color: #6c757d;">Prediction Confidence</div>',
                                '</div>'
                            ])
                        except Exception as e:
                            print(f"Error calculating confidence: {str(e)}")
                    
                    dashboard_parts.append('</div>')
            except Exception as e:
                print(f"Error generating confidence section: {str(e)}")
            
            # Placeholder for future visualizations
            dashboard_parts.extend([
                '<div style="margin: 20px 0; padding: 15px; background: #e9ecef; border-radius: 5px; text-align: center;">',
                '<h3 style="color: #6c757d;">Interactive Visualizations</h3>',
                '<p style="color: #6c757d;">Advanced SHAP plots (waterfall, dependence, interaction) would be embedded here.</p>',
                '<p style="color: #6c757d;">These would include:</p>',
                '<ul style="color: #6c757d; text-align: left; max-width: 400px; margin: 0 auto;">',
                '<li>Waterfall plots for individual predictions</li>',
                '<li>Feature dependence plots</li>',
                '<li>Feature interaction heatmaps</li>',
                '<li>Summary plots across multiple predictions</li>',
                '</ul>',
                '</div>',
                '</div>'
            ])
            
            return '\n'.join(dashboard_parts)
            
        except Exception as e:
            return f'<div style="color: #dc3545; padding: 20px;">Error generating SHAP dashboard: {str(e)}</div>'
    
    def get_shap_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics from SHAP analysis.
        """
        try:
            stats = {
                'total_explanations': 0,
                'features_analyzed': 0,
                'avg_confidence': 0.0,
                'top_features': [],
                'explanation_dates': []
            }
            
            if hasattr(self.strategy, 'explanation_cache') and self.strategy.explanation_cache:
                stats['total_explanations'] = len(self.strategy.explanation_cache)
                stats['explanation_dates'] = list(self.strategy.explanation_cache.keys())
                
                # Get feature statistics
                if hasattr(self.strategy, 'get_feature_importance'):
                    feature_importance = self.strategy.get_feature_importance()
                    if feature_importance:
                        stats['features_analyzed'] = len(feature_importance)
                        stats['top_features'] = list(feature_importance.keys())[:5]
                
                # Calculate average confidence
                confidences = []
                for explanation in self.strategy.explanation_cache.values():
                    shap_values = explanation.get('shap_values', [])
                    if self._has_values(shap_values):
                        try:
                            if isinstance(shap_values, (list, tuple)):
                                total_mag = sum(float(np.mean(np.abs(sv))) for sv in shap_values if sv is not None)
                            else:
                                total_mag = float(np.mean(np.abs(shap_values)))
                            confidences.append(min(total_mag, 1.0))
                        except Exception:
                            continue
                
                if confidences:
                    stats['avg_confidence'] = np.mean(confidences)
            
            return stats
            
        except Exception as e:
            print(f"Error getting SHAP summary stats: {str(e)}")
            return {'error': str(e)}
