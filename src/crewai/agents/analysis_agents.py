"""
Analysis Agents for CrewAI Investment System

This module contains analysis agents responsible for technical, fundamental,
sentiment, and valuation analysis.
"""

from typing import List, Any, Dict, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import json

from crewai.tools import BaseTool
from src.tools.api import prices_to_df
from src.utils.logging_config import setup_logger

from .base import AnalysisAgent
from ..config.state import InvestmentState


logger = setup_logger('crewai_analysis_agents')


class TechnicalAnalysisTool(BaseTool):
    """Tool for technical analysis calculations"""
    name: str = "calculate_technical_indicators"
    description: str = "Calculate technical indicators like MACD, RSI, Bollinger Bands, and generate trading signals"

    def _run(self, price_data: List[Dict[str, float]]) -> Dict[str, Any]:
        """Calculate technical indicators from price data"""
        try:
            prices_df = prices_to_df(price_data)
            if prices_df.empty:
                return {"error": "No price data available"}

            # Calculate technical indicators
            indicators = self._calculate_all_indicators(prices_df)
            signals = self._generate_signals(indicators, prices_df)

            return {
                "indicators": indicators,
                "signals": signals,
                "current_price": prices_df['close'].iloc[-1],
                "analysis_time": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Technical analysis failed: {str(e)}")
            return {"error": str(e)}

    def _calculate_all_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all technical indicators"""
        indicators = {}

        # MACD
        macd_line, signal_line = self._calculate_macd(df)
        indicators['macd'] = {
            'macd_line': macd_line.iloc[-1] if not macd_line.empty else None,
            'signal_line': signal_line.iloc[-1] if not signal_line.empty else None,
            'histogram': macd_line.iloc[-1] - signal_line.iloc[-1] if not macd_line.empty and not signal_line.empty else None
        }

        # RSI
        rsi = self._calculate_rsi(df)
        indicators['rsi'] = rsi.iloc[-1] if not rsi.empty else None

        # Bollinger Bands
        upper_band, lower_band = self._calculate_bollinger_bands(df)
        indicators['bollinger_bands'] = {
            'upper': upper_band.iloc[-1] if not upper_band.empty else None,
            'middle': (upper_band + lower_band).iloc[-1] / 2 if not upper_band.empty and not lower_band.empty else None,
            'lower': lower_band.iloc[-1] if not lower_band.empty else None
        }

        # OBV (On-Balance Volume)
        obv = self._calculate_obv(df)
        obv_slope = obv.diff().iloc[-5:].mean() if len(obv) >= 5 else 0
        indicators['obv'] = {
            'current': obv.iloc[-1] if not obv.empty else None,
            'slope': obv_slope
        }

        # Moving averages
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_50'] = df['close'].rolling(window=50).mean()
        indicators['moving_averages'] = {
            'ma_20': df['ma_20'].iloc[-1] if not df['ma_20'].empty else None,
            'ma_50': df['ma_50'].iloc[-1] if not df['ma_50'].empty else None
        }

        # Price momentum
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)
        indicators['momentum'] = {
            'momentum_5': df['momentum_5'].iloc[-1] if not df['momentum_5'].empty else None,
            'momentum_10': df['momentum_10'].iloc[-1] if not df['momentum_10'].empty else None
        }

        return indicators

    def _generate_signals(self, indicators: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals from indicators"""
        signals = {
            'individual_signals': [],
            'overall_signal': 'neutral',
            'confidence': 0.0,
            'reasoning': []
        }

        individual_signals = []

        # MACD signal
        macd_data = indicators.get('macd', {})
        if macd_data.get('macd_line') and macd_data.get('signal_line'):
            macd_line = macd_data['macd_line']
            signal_line = macd_data['signal_line']

            # Get previous values for crossover detection
            prev_macd = None
            prev_signal = None
            if len(df) >= 2:
                prev_macd_line, prev_signal_line = self._calculate_macd(df.iloc[:-1])
                if not prev_macd_line.empty and not prev_signal_line.empty:
                    prev_macd = prev_macd_line.iloc[-1]
                    prev_signal = prev_signal_line.iloc[-1]

            if prev_macd and prev_signal:
                if prev_macd < prev_signal and macd_line > signal_line:
                    individual_signals.append({'indicator': 'MACD', 'signal': 'bullish', 'reason': 'Bullish crossover'})
                elif prev_macd > prev_signal and macd_line < signal_line:
                    individual_signals.append({'indicator': 'MACD', 'signal': 'bearish', 'reason': 'Bearish crossover'})
                else:
                    individual_signals.append({'indicator': 'MACD', 'signal': 'neutral', 'reason': 'No clear signal'})

        # RSI signal
        rsi = indicators.get('rsi')
        if rsi is not None:
            if rsi < 30:
                individual_signals.append({'indicator': 'RSI', 'signal': 'bullish', 'reason': 'Oversold condition'})
            elif rsi > 70:
                individual_signals.append({'indicator': 'RSI', 'signal': 'bearish', 'reason': 'Overbought condition'})
            else:
                individual_signals.append({'indicator': 'RSI', 'signal': 'neutral', 'reason': 'Neutral territory'})

        # Bollinger Bands signal
        bb_data = indicators.get('bollinger_bands', {})
        if bb_data.get('upper') and bb_data.get('lower'):
            current_price = df['close'].iloc[-1]
            upper_band = bb_data['upper']
            lower_band = bb_data['lower']

            if current_price < lower_band:
                individual_signals.append({'indicator': 'Bollinger Bands', 'signal': 'bullish', 'reason': 'Price below lower band'})
            elif current_price > upper_band:
                individual_signals.append({'indicator': 'Bollinger Bands', 'signal': 'bearish', 'reason': 'Price above upper band'})
            else:
                individual_signals.append({'indicator': 'Bollinger Bands', 'signal': 'neutral', 'reason': 'Price within bands'})

        # Moving averages signal
        ma_data = indicators.get('moving_averages', {})
        if ma_data.get('ma_20') and ma_data.get('ma_50'):
            ma_20 = ma_data['ma_20']
            ma_50 = ma_data['ma_50']

            if ma_20 > ma_50:
                individual_signals.append({'indicator': 'Moving Averages', 'signal': 'bullish', 'reason': 'Golden cross pattern'})
            elif ma_20 < ma_50:
                individual_signals.append({'indicator': 'Moving Averages', 'signal': 'bearish', 'reason': 'Death cross pattern'})
            else:
                individual_signals.append({'indicator': 'Moving Averages', 'signal': 'neutral', 'reason': 'Neutral position'})

        # OBV signal
        obv_data = indicators.get('obv', {})
        if obv_data.get('slope') is not None:
            obv_slope = obv_data['slope']
            if obv_slope > 0:
                individual_signals.append({'indicator': 'OBV', 'signal': 'bullish', 'reason': 'Positive volume trend'})
            elif obv_slope < 0:
                individual_signals.append({'indicator': 'OBV', 'signal': 'bearish', 'reason': 'Negative volume trend'})
            else:
                individual_signals.append({'indicator': 'OBV', 'signal': 'neutral', 'reason': 'Neutral volume trend'})

        # Calculate overall signal
        signals['individual_signals'] = individual_signals

        # Count bullish and bearish signals
        bullish_count = sum(1 for s in individual_signals if s['signal'] == 'bullish')
        bearish_count = sum(1 for s in individual_signals if s['signal'] == 'bearish')
        total_signals = len(individual_signals)

        if total_signals > 0:
            if bullish_count > bearish_count:
                signals['overall_signal'] = 'bullish'
                signals['confidence'] = bullish_count / total_signals
            elif bearish_count > bullish_count:
                signals['overall_signal'] = 'bearish'
                signals['confidence'] = bearish_count / total_signals
            else:
                signals['overall_signal'] = 'neutral'
                signals['confidence'] = 0.5

        signals['reasoning'] = [f"{s['indicator']}: {s['reason']}" for s in individual_signals]

        return signals

    def _calculate_macd(self, df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """Calculate MACD"""
        if len(df) < slow_period + signal_period:
            return pd.Series(), pd.Series()

        exp1 = df['close'].ewm(span=fast_period).mean()
        exp2 = df['close'].ewm(span=slow_period).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal_period).mean()

        return macd_line, signal_line

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14):
        """Calculate RSI"""
        if len(df) < period + 1:
            return pd.Series()

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: int = 2):
        """Calculate Bollinger Bands"""
        if len(df) < period:
            return pd.Series(), pd.Series()

        middle_band = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)

        return upper_band, lower_band

    def _calculate_obv(self, df: pd.DataFrame):
        """Calculate On-Balance Volume"""
        obv = np.where(df['close'] > df['close'].shift(1), df['volume'],
                      np.where(df['close'] < df['close'].shift(1), -df['volume'], 0))
        return pd.Series(obv, index=df.index).cumsum()


class FundamentalAnalysisTool(BaseTool):
    """Tool for fundamental analysis"""
    name: str = "analyze_fundamentals"
    description: str = "Analyze fundamental metrics and financial health of a company"

    def _run(self, financial_metrics: Dict[str, Any], financial_statements: Dict[str, Any]) -> Dict[str, Any]:
        """Perform fundamental analysis"""
        try:
            analysis = {
                'financial_health': self._analyze_financial_health(financial_metrics),
                'profitability': self._analyze_profitability(financial_metrics),
                'growth': self._analyze_growth(financial_metrics),
                'valuation_ratios': self._analyze_valuation_ratios(financial_metrics),
                'overall_assessment': 'neutral',
                'confidence': 0.0,
                'key_metrics': financial_metrics
            }

            # Calculate overall assessment
            scores = []
            if analysis['financial_health']['score'] is not None:
                scores.append(analysis['financial_health']['score'])
            if analysis['profitability']['score'] is not None:
                scores.append(analysis['profitability']['score'])
            if analysis['growth']['score'] is not None:
                scores.append(analysis['growth']['score'])

            if scores:
                avg_score = np.mean(scores)
                if avg_score > 0.6:
                    analysis['overall_assessment'] = 'bullish'
                elif avg_score < 0.4:
                    analysis['overall_assessment'] = 'bearish'
                else:
                    analysis['overall_assessment'] = 'neutral'

                analysis['confidence'] = abs(avg_score - 0.5) * 2  # Convert to 0-1 scale

            return analysis

        except Exception as e:
            logger.error(f"Fundamental analysis failed: {str(e)}")
            return {"error": str(e)}

    def _analyze_financial_health(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze financial health"""
        analysis = {
            'assessment': 'neutral',
            'score': None,
            'factors': []
        }

        try:
            # Current ratio (liquidity)
            current_ratio = metrics.get('current_ratio')
            if current_ratio:
                if current_ratio > 2.0:
                    analysis['factors'].append('Strong liquidity position')
                    analysis['score'] = 0.7
                elif current_ratio > 1.0:
                    analysis['factors'].append('Adequate liquidity')
                    analysis['score'] = 0.5
                else:
                    analysis['factors'].append('Weak liquidity position')
                    analysis['score'] = 0.3

            # Debt to equity
            debt_to_equity = metrics.get('debt_to_equity')
            if debt_to_equity:
                if debt_to_equity < 0.3:
                    analysis['factors'].append('Low debt levels')
                    analysis['score'] = (analysis['score'] or 0.5) + 0.1
                elif debt_to_equity > 0.8:
                    analysis['factors'].append('High debt levels')
                    analysis['score'] = (analysis['score'] or 0.5) - 0.1

        except Exception as e:
            logger.error(f"Financial health analysis failed: {str(e)}")

        return analysis

    def _analyze_profitability(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze profitability"""
        analysis = {
            'assessment': 'neutral',
            'score': None,
            'factors': []
        }

        try:
            # Return on Equity
            roe = metrics.get('return_on_equity')
            if roe:
                if roe > 0.15:  # 15%
                    analysis['factors'].append('Strong ROE')
                    analysis['score'] = 0.7
                elif roe > 0.08:  # 8%
                    analysis['factors'].append('Moderate ROE')
                    analysis['score'] = 0.5
                else:
                    analysis['factors'].append('Weak ROE')
                    analysis['score'] = 0.3

            # Net margin
            net_margin = metrics.get('net_margin')
            if net_margin:
                if net_margin > 0.15:  # 15%
                    analysis['factors'].append('Strong net margins')
                    analysis['score'] = (analysis['score'] or 0.5) + 0.1
                elif net_margin < 0.05:  # 5%
                    analysis['factors'].append('Weak net margins')
                    analysis['score'] = (analysis['score'] or 0.5) - 0.1

        except Exception as e:
            logger.error(f"Profitability analysis failed: {str(e)}")

        return analysis

    def _analyze_growth(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze growth metrics"""
        analysis = {
            'assessment': 'neutral',
            'score': None,
            'factors': []
        }

        try:
            # Revenue growth
            revenue_growth = metrics.get('revenue_growth')
            if revenue_growth:
                if revenue_growth > 0.20:  # 20%
                    analysis['factors'].append('Strong revenue growth')
                    analysis['score'] = 0.7
                elif revenue_growth > 0.10:  # 10%
                    analysis['factors'].append('Moderate revenue growth')
                    analysis['score'] = 0.5
                else:
                    analysis['factors'].append('Weak revenue growth')
                    analysis['score'] = 0.3

            # Earnings growth
            earnings_growth = metrics.get('earnings_growth')
            if earnings_growth:
                if earnings_growth > 0.20:  # 20%
                    analysis['factors'].append('Strong earnings growth')
                    analysis['score'] = (analysis['score'] or 0.5) + 0.1
                elif earnings_growth < 0:
                    analysis['factors'].append('Negative earnings growth')
                    analysis['score'] = (analysis['score'] or 0.5) - 0.1

        except Exception as e:
            logger.error(f"Growth analysis failed: {str(e)}")

        return analysis

    def _analyze_valuation_ratios(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze valuation ratios"""
        analysis = {
            'assessment': 'neutral',
            'score': None,
            'factors': []
        }

        try:
            # PE ratio
            pe_ratio = metrics.get('pe_ratio')
            if pe_ratio:
                if pe_ratio < 15:  # Reasonable valuation
                    analysis['factors'].append('Attractive PE ratio')
                    analysis['score'] = 0.6
                elif pe_ratio > 30:  # Expensive
                    analysis['factors'].append('High PE ratio')
                    analysis['score'] = 0.4

            # PB ratio
            pb_ratio = metrics.get('price_to_book')
            if pb_ratio:
                if pb_ratio < 1.5:  # Reasonable
                    analysis['factors'].append('Attractive PB ratio')
                    analysis['score'] = (analysis['score'] or 0.5) + 0.1
                elif pb_ratio > 3:  # Expensive
                    analysis['factors'].append('High PB ratio')
                    analysis['score'] = (analysis['score'] or 0.5) - 0.1

        except Exception as e:
            logger.error(f"Valuation analysis failed: {str(e)}")

        return analysis


class TechnicalAnalysisAgent(AnalysisAgent):
    """
    Technical Analysis Agent responsible for analyzing price trends,
    technical indicators, and generating trading signals.
    """

    def __init__(self, tools: List[BaseTool] = None):
        super().__init__(
            role="Senior Technical Analyst",
            goal="Analyze price trends, technical indicators, and market patterns to generate accurate trading signals",
            backstory="""You are a veteran technical analyst with 15+ years of experience in chart analysis,
            technical indicator interpretation, and market timing. You specialize in multiple analysis methods
            including trend following, momentum trading, mean reversion, and statistical arbitrage.
            You have deep expertise in indicators like MACD, RSI, Bollinger Bands, and volume analysis.
            You understand both classical chart patterns and modern quantitative techniques.""",
            tools=tools or [TechnicalAnalysisTool()],
            analysis_type="technical"
        )

    def create_task_description(self, state: InvestmentState) -> str:
        """Create task description for technical analysis"""
        price_data = state.data_cache.get('price_history', [])

        return f"""
        Perform comprehensive technical analysis for {state.ticker} using available price data.

        Your analysis should include:
        1. Trend analysis (moving averages, trend lines)
        2. Momentum indicators (MACD, RSI, Stochastic)
        3. Volatility analysis (Bollinger Bands, ATR)
        4. Volume analysis (OBV, volume patterns)
        5. Support and resistance levels
        6. Chart pattern recognition
        7. Trading signal generation

        Available data: {len(price_data)} price points from {state.start_date} to {state.end_date}

        Requirements:
        - Calculate key technical indicators
        - Generate clear trading signals (bullish/bearish/neutral)
        - Provide confidence levels for each signal
        - Explain reasoning behind each signal
        - Identify key support/resistance levels
        - Assess overall market trend

        Expected Output:
        A comprehensive technical analysis report with trading signals, confidence levels,
        and detailed reasoning for {state.ticker}.
        """

    def postprocess_output(self, output: Any, state: InvestmentState) -> Any:
        """Postprocess technical analysis output"""
        try:
            if isinstance(output, str):
                try:
                    output = json.loads(output)
                except json.JSONDecodeError:
                    # Parse structured output from text
                    output = self._parse_text_output(output)

            if isinstance(output, dict):
                state.data_cache['technical_analysis'] = output
                state.update_analysis_result('technical_analysis', output)

            return output

        except Exception as e:
            logger.error(f"Failed to postprocess technical analysis: {str(e)}")
            error_output = {"error": str(e), "analysis_type": "technical"}
            state.update_analysis_result('technical_analysis', error_output)
            return error_output

    def _parse_text_output(self, text_output: str) -> Dict[str, Any]:
        """Parse text output into structured format"""
        # This is a simplified parser - in practice, you'd use more sophisticated NLP
        lines = text_output.split('\n')
        result = {
            "signal": "neutral",
            "confidence": 0.5,
            "reasoning": text_output,
            "analysis_time": datetime.now().isoformat()
        }

        # Simple keyword-based signal extraction
        text_lower = text_output.lower()
        if 'bullish' in text_lower or 'buy' in text_lower:
            result['signal'] = 'bullish'
            result['confidence'] = 0.7
        elif 'bearish' in text_lower or 'sell' in text_lower:
            result['signal'] = 'bearish'
            result['confidence'] = 0.7

        return result


class FundamentalAnalysisAgent(AnalysisAgent):
    """
    Fundamental Analysis Agent responsible for analyzing financial metrics,
    company health, and intrinsic value.
    """

    def __init__(self, tools: List[BaseTool] = None):
        super().__init__(
            role="Senior Fundamental Analyst",
            goal="Analyze financial metrics, business health, and intrinsic value to determine long-term investment potential",
            backstory="""You are a seasoned fundamental analyst with deep expertise in financial statement analysis,
            business valuation, and competitive analysis. You have 15+ years of experience evaluating companies
            across different industries. You specialize in identifying durable competitive advantages,
            understanding business models, and assessing management quality. You are known for your
            thorough due diligence and conservative valuation approach.""",
            tools=tools or [FundamentalAnalysisTool()],
            analysis_type="fundamental"
        )

    def create_task_description(self, state: InvestmentState) -> str:
        """Create task description for fundamental analysis"""
        financial_metrics = state.data_cache.get('financial_metrics', {})
        financial_statements = state.data_cache.get('financial_statements', {})

        return f"""
        Perform comprehensive fundamental analysis for {state.ticker} using available financial data.

        Your analysis should cover:
        1. Business model and competitive position
        2. Financial health (liquidity, solvency, efficiency)
        3. Profitability analysis (margins, returns, efficiency ratios)
        4. Growth trends and sustainability
        5. Management quality and corporate governance
        6. Industry position and market dynamics
        7. Valuation assessment and intrinsic value

        Available data:
        - Financial metrics: {list(financial_metrics.keys()) if financial_metrics else 'None'}
        - Financial statements available: {bool(financial_statements)}

        Requirements:
        - Assess overall financial health
        - Evaluate competitive advantages
        - Analyze growth prospects
        - Determine intrinsic value
        - Provide investment recommendation (buy/hold/sell)
        - Explain reasoning with specific metrics

        Expected Output:
        A comprehensive fundamental analysis report with investment recommendation,
        valuation assessment, and detailed reasoning for {state.ticker}.
        """

    def postprocess_output(self, output: Any, state: InvestmentState) -> Any:
        """Postprocess fundamental analysis output"""
        try:
            if isinstance(output, str):
                try:
                    output = json.loads(output)
                except json.JSONDecodeError:
                    output = self._parse_text_output(output)

            if isinstance(output, dict):
                state.data_cache['fundamental_analysis'] = output
                state.update_analysis_result('fundamental_analysis', output)

            return output

        except Exception as e:
            logger.error(f"Failed to postprocess fundamental analysis: {str(e)}")
            error_output = {"error": str(e), "analysis_type": "fundamental"}
            state.update_analysis_result('fundamental_analysis', error_output)
            return error_output

    def _parse_text_output(self, text_output: str) -> Dict[str, Any]:
        """Parse text output into structured format"""
        result = {
            "signal": "neutral",
            "confidence": 0.5,
            "reasoning": text_output,
            "analysis_time": datetime.now().isoformat()
        }

        # Simple keyword-based signal extraction
        text_lower = text_output.lower()
        if 'strong buy' in text_lower or 'undervalued' in text_lower:
            result['signal'] = 'bullish'
            result['confidence'] = 0.8
        elif 'buy' in text_lower or 'attractive' in text_lower:
            result['signal'] = 'bullish'
            result['confidence'] = 0.6
        elif 'sell' in text_lower or 'overvalued' in text_lower:
            result['signal'] = 'bearish'
            result['confidence'] = 0.7
        elif 'strong sell' in text_lower or 'avoid' in text_lower:
            result['signal'] = 'bearish'
            result['confidence'] = 0.9

        return result


# Sentiment and Valuation agents will be added in the next iteration
# due to length constraints