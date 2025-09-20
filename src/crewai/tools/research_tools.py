"""
Research Tools for CrewAI Investment System

This module provides specialized tools for investment research, debate moderation,
and perspective analysis.
"""

from typing import Dict, Any, List, Optional
from crewai.tools import BaseTool
from datetime import datetime
import json
import logging

from ..config.state import InvestmentState

logger = logging.getLogger(__name__)


class BullishResearchTool(BaseTool):
    """
    Tool for conducting bullish investment research and analysis
    """

    name: str = "bullish_research_tool"
    description: str = """
    Analyzes investment opportunities from an optimistic perspective to identify
    growth potential and construct bullish investment theses.

    Input should be a JSON string containing:
    - ticker: Stock symbol
    - analysis_data: Dictionary with technical, fundamental, sentiment, and valuation results
    - current_portfolio: Current portfolio state
    - analysis_period: Time period for analysis

    Returns bullish research analysis with arguments, confidence levels, and growth catalysts.
    """

    def _run(self, input_data: str) -> str:
        """Execute bullish research analysis"""
        try:
            # Parse input data
            data = json.loads(input_data)
            ticker = data.get('ticker', 'UNKNOWN')
            analysis_data = data.get('analysis_data', {})
            portfolio = data.get('current_portfolio', {})
            period = data.get('analysis_period', {})

            logger.info(f"Conducting bullish research for {ticker}")

            # Extract analysis results
            technical_data = analysis_data.get('technical_analysis_result', {})
            fundamental_data = analysis_data.get('fundamental_analysis_result', {})
            sentiment_data = analysis_data.get('news_analysis_result', {})
            valuation_data = analysis_data.get('valuation_analysis_result', {})

            # Generate bullish arguments
            bullish_points = []
            confidence_scores = []

            # Technical analysis from bullish perspective
            if technical_data:
                tech_signal = technical_data.get('signal', 'neutral')
                tech_confidence = technical_data.get('confidence', 0.5)

                if tech_signal == 'bullish':
                    bullish_points.append(f"Technical indicators show strong bullish momentum with {tech_confidence:.1%} confidence")
                    confidence_scores.append(tech_confidence)
                else:
                    bullish_points.append("Current technical weakness may present attractive entry opportunities for long-term investors")
                    confidence_scores.append(0.4)

                # Add specific technical indicators
                indicators = technical_data.get('indicators', {})
                if indicators.get('rsi', 50) < 30:
                    bullish_points.append("RSI indicates oversold conditions, suggesting potential rebound opportunity")
                    confidence_scores.append(0.6)

            # Fundamental analysis from bullish perspective
            if fundamental_data:
                fund_signal = fundamental_data.get('signal', 'neutral')
                fund_confidence = fundamental_data.get('confidence', 0.5)

                if fund_signal == 'bullish':
                    bullish_points.append(f"Strong fundamentals support growth thesis with {fund_confidence:.1%} confidence")
                    confidence_scores.append(fund_confidence)
                else:
                    bullish_points.append("Current fundamental challenges may be temporary and represent turnaround potential")
                    confidence_scores.append(0.4)

                # Add specific fundamental factors
                metrics = fundamental_data.get('metrics', {})
                if metrics.get('revenue_growth', 0) > 0.1:
                    bullish_points.append(f"Strong revenue growth of {metrics['revenue_growth']:.1%} indicates expanding business")
                    confidence_scores.append(0.7)

            # Sentiment analysis from bullish perspective
            if sentiment_data:
                sent_signal = sentiment_data.get('signal', 'neutral')
                sent_confidence = sentiment_data.get('confidence', 0.5)

                if sent_signal == 'bullish':
                    bullish_points.append(f"Positive market sentiment supports bullish outlook with {sent_confidence:.1%} confidence")
                    confidence_scores.append(sent_confidence)
                else:
                    bullish_points.append("Current negative sentiment may be overdone, creating contrarian opportunity")
                    confidence_scores.append(0.4)

            # Valuation analysis from bullish perspective
            if valuation_data:
                val_signal = valuation_data.get('signal', 'neutral')
                val_confidence = valuation_data.get('confidence', 0.5)

                if val_signal == 'bullish':
                    bullish_points.append(f"Attractive valuation with {val_confidence:.1%} confidence suggests upside potential")
                    confidence_scores.append(val_confidence)
                else:
                    bullish_points.append("Premium valuation justified by superior growth prospects and quality")
                    confidence_scores.append(0.4)

            # Calculate overall confidence
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5

            # Identify growth catalysts
            catalysts = self._identify_growth_catalysts(analysis_data)

            # Generate risk factors that could undermine bullish thesis
            risk_factors = [
                "Market volatility and economic uncertainty",
                "Competitive pressures and industry disruption",
                "Regulatory changes and compliance requirements",
                "Execution risk in business strategy"
            ]

            # Construct result
            result = {
                "perspective": "bullish",
                "ticker": ticker,
                "confidence": avg_confidence,
                "thesis_points": bullish_points,
                "growth_catalysts": catalysts,
                "risk_factors": risk_factors,
                "analysis_timestamp": datetime.now().isoformat(),
                "reasoning": f"Bullish thesis for {ticker} based on comprehensive analysis of technical, fundamental, sentiment, and valuation factors from an optimistic perspective"
            }

            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error in bullish research: {str(e)}")
            return json.dumps({
                "error": str(e),
                "perspective": "bullish",
                "confidence": 0.0,
                "thesis_points": ["Analysis failed due to technical error"],
                "timestamp": datetime.now().isoformat()
            })

    def _identify_growth_catalysts(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Identify potential growth catalysts from analysis data"""
        catalysts = []

        # Technical catalysts
        technical = analysis_data.get('technical_analysis_result', {})
        if technical.get('signal') == 'bullish':
            catalysts.append("Technical breakout potential with positive momentum indicators")

        # Fundamental catalysts
        fundamental = analysis_data.get('fundamental_analysis_result', {})
        metrics = fundamental.get('metrics', {})
        if metrics.get('roe', 0) > 0.15:
            catalysts.append("Strong return on equity indicates efficient capital allocation")
        if metrics.get('debt_to_equity', 1) < 0.5:
            catalysts.append("Healthy balance sheet provides flexibility for growth investments")

        # Market catalysts
        catalysts.extend([
            "Industry growth trends and market expansion opportunities",
            "Potential strategic partnerships and M&A activities",
            "Innovation pipeline and product development initiatives"
        ])

        return catalysts[:5]  # Return top 5 catalysts


class BearishResearchTool(BaseTool):
    """
    Tool for conducting bearish investment research and risk analysis
    """

    name: str = "bearish_research_tool"
    description: str = """
    Analyzes investment opportunities from a cautious perspective to identify
    risks, threats, and potential downside factors.

    Input should be a JSON string containing:
    - ticker: Stock symbol
    - analysis_data: Dictionary with technical, fundamental, sentiment, and valuation results
    - current_portfolio: Current portfolio state
    - analysis_period: Time period for analysis

    Returns bearish risk assessment with threats, confidence levels, and risk factors.
    """

    def _run(self, input_data: str) -> str:
        """Execute bearish research analysis"""
        try:
            # Parse input data
            data = json.loads(input_data)
            ticker = data.get('ticker', 'UNKNOWN')
            analysis_data = data.get('analysis_data', {})
            portfolio = data.get('current_portfolio', {})
            period = data.get('analysis_period', {})

            logger.info(f"Conducting bearish research for {ticker}")

            # Extract analysis results
            technical_data = analysis_data.get('technical_analysis_result', {})
            fundamental_data = analysis_data.get('fundamental_analysis_result', {})
            sentiment_data = analysis_data.get('news_analysis_result', {})
            valuation_data = analysis_data.get('valuation_analysis_result', {})

            # Generate bearish arguments
            bearish_points = []
            confidence_scores = []

            # Technical analysis from bearish perspective
            if technical_data:
                tech_signal = technical_data.get('signal', 'neutral')
                tech_confidence = technical_data.get('confidence', 0.5)

                if tech_signal == 'bearish':
                    bearish_points.append(f"Technical indicators show bearish momentum with {tech_confidence:.1%} confidence")
                    confidence_scores.append(tech_confidence)
                else:
                    bearish_points.append("Current technical strength may be unsustainable and prone to reversal")
                    confidence_scores.append(0.4)

                # Add specific technical indicators
                indicators = technical_data.get('indicators', {})
                if indicators.get('rsi', 50) > 70:
                    bearish_points.append("RSI indicates overbought conditions, suggesting potential pullback risk")
                    confidence_scores.append(0.6)

            # Fundamental analysis from bearish perspective
            if fundamental_data:
                fund_signal = fundamental_data.get('signal', 'neutral')
                fund_confidence = fundamental_data.get('confidence', 0.5)

                if fund_signal == 'bearish':
                    bearish_points.append(f"Weak fundamentals present significant risks with {fund_confidence:.1%} confidence")
                    confidence_scores.append(fund_confidence)
                else:
                    bearish_points.append("Current fundamental strength may deteriorate due to market pressures")
                    confidence_scores.append(0.4)

                # Add specific fundamental factors
                metrics = fundamental_data.get('metrics', {})
                if metrics.get('debt_to_equity', 0) > 1.0:
                    bearish_points.append(f"High debt-to-equity ratio of {metrics['debt_to_equity']:.2f} indicates financial risk")
                    confidence_scores.append(0.7)

            # Sentiment analysis from bearish perspective
            if sentiment_data:
                sent_signal = sentiment_data.get('signal', 'neutral')
                sent_confidence = sentiment_data.get('confidence', 0.5)

                if sent_signal == 'bearish':
                    bearish_points.append(f"Negative market sentiment highlights risks with {sent_confidence:.1%} confidence")
                    confidence_scores.append(sent_confidence)
                else:
                    bearish_points.append("Current positive sentiment may be complacent and vulnerable to shocks")
                    confidence_scores.append(0.4)

            # Valuation analysis from bearish perspective
            if valuation_data:
                val_signal = valuation_data.get('signal', 'neutral')
                val_confidence = valuation_data.get('confidence', 0.5)

                if val_signal == 'bearish':
                    bearish_points.append(f"Overvalued with {val_confidence:.1%} confidence suggests downside risk")
                    confidence_scores.append(val_confidence)
                else:
                    bearish_points.append("Current reasonable valuation may not compensate for hidden risks")
                    confidence_scores.append(0.4)

            # Calculate overall confidence
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5

            # Identify risk factors
            risk_factors = self._identify_risk_factors(analysis_data)

            # Generate mitigating factors that could reduce bearish concerns
            mitigating_factors = [
                "Strong management team with proven track record",
                "Resilient business model with diversified revenue streams",
                "Adequate liquidity and financial flexibility",
                "Favorable industry long-term trends"
            ]

            # Construct result
            result = {
                "perspective": "bearish",
                "ticker": ticker,
                "confidence": avg_confidence,
                "thesis_points": bearish_points,
                "risk_factors": risk_factors,
                "mitigating_factors": mitigating_factors,
                "analysis_timestamp": datetime.now().isoformat(),
                "reasoning": f"Bearish risk assessment for {ticker} based on comprehensive analysis of technical, fundamental, sentiment, and valuation factors from a cautious perspective"
            }

            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error in bearish research: {str(e)}")
            return json.dumps({
                "error": str(e),
                "perspective": "bearish",
                "confidence": 0.0,
                "thesis_points": ["Analysis failed due to technical error"],
                "timestamp": datetime.now().isoformat()
            })

    def _identify_risk_factors(self, analysis_data: Dict[str, Any]) -> List[str]:
        """Identify potential risk factors from analysis data"""
        risk_factors = []

        # Technical risks
        technical = analysis_data.get('technical_analysis_result', {})
        if technical.get('signal') == 'bearish':
            risk_factors.append("Technical breakdown potential with negative momentum indicators")

        # Fundamental risks
        fundamental = analysis_data.get('fundamental_analysis_result', {})
        metrics = fundamental.get('metrics', {})
        if metrics.get('roe', 0) < 0.08:
            risk_factors.append("Low return on equity indicates inefficient capital allocation")
        if metrics.get('debt_to_equity', 0) > 1.0:
            risk_factors.append("High leverage increases financial risk and bankruptcy potential")

        # Market risks
        risk_factors.extend([
            "Economic downturn and recession risks",
            "Industry disruption and technological obsolescence",
            "Regulatory changes and compliance costs",
            "Competitive pressures and margin compression"
        ])

        return risk_factors[:5]  # Return top 5 risk factors


class DebateModerationTool(BaseTool):
    """
    Tool for moderating investment debates and synthesizing diverse perspectives
    """

    name: str = "debate_moderation_tool"
    description: str = """
    Facilitates balanced investment debates between different perspectives and
    synthesizes competing viewpoints into coherent conclusions.

    Input should be a JSON string containing:
    - ticker: Stock symbol
    - bullish_thesis: Bullish research results with arguments and confidence
    - bearish_thesis: Bearish research results with risks and concerns
    - analysis_context: Additional context for the debate
    - current_portfolio: Current portfolio state

    Returns balanced debate analysis with final recommendation and reasoning.
    """

    def _run(self, input_data: str) -> str:
        """Execute debate moderation and synthesis"""
        try:
            # Parse input data
            data = json.loads(input_data)
            ticker = data.get('ticker', 'UNKNOWN')
            bullish_thesis = data.get('bullish_thesis', {})
            bearish_thesis = data.get('bearish_thesis', {})
            context = data.get('analysis_context', {})
            portfolio = data.get('current_portfolio', {})

            logger.info(f"Moderating investment debate for {ticker}")

            # Extract thesis information
            bull_confidence = bullish_thesis.get('confidence', 0.0)
            bear_confidence = bearish_thesis.get('confidence', 0.0)
            bull_points = bullish_thesis.get('thesis_points', [])
            bear_points = bearish_thesis.get('thesis_points', [])

            # Calculate confidence differential
            confidence_diff = bull_confidence - bear_confidence

            # Generate debate summary
            debate_summary = []
            debate_summary.append("BULLISH ARGUMENTS:")
            for i, point in enumerate(bull_points, 1):
                debate_summary.append(f"{i}. {point}")

            debate_summary.append("\nBEARISH ARGUMENTS:")
            for i, point in enumerate(bear_points, 1):
                debate_summary.append(f"{i}. {point}")

            # Assess argument strength
            argument_assessment = self._assess_arguments(bullish_thesis, bearish_thesis)

            # Determine final signal based on debate
            final_signal, reasoning, final_confidence = self._determine_final_signal(
                bull_confidence, bear_confidence, confidence_diff, argument_assessment
            )

            # Generate investment recommendation
            recommendation = self._generate_recommendation(
                final_signal, final_confidence, ticker, portfolio
            )

            # Construct result
            result = {
                "ticker": ticker,
                "final_signal": final_signal,
                "confidence": final_confidence,
                "bull_confidence": bull_confidence,
                "bear_confidence": bear_confidence,
                "confidence_differential": confidence_diff,
                "debate_summary": debate_summary,
                "argument_assessment": argument_assessment,
                "reasoning": reasoning,
                "recommendation": recommendation,
                "moderation_timestamp": datetime.now().isoformat()
            }

            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Error in debate moderation: {str(e)}")
            return json.dumps({
                "error": str(e),
                "final_signal": "neutral",
                "confidence": 0.5,
                "reasoning": "Debate analysis failed due to technical error",
                "timestamp": datetime.now().isoformat()
            })

    def _assess_arguments(self, bullish_thesis: Dict, bearish_thesis: Dict) -> Dict[str, Any]:
        """Assess the strength and credibility of arguments from both sides"""
        assessment = {
            "bullish_strength": 0.0,
            "bearish_strength": 0.0,
            "key_bullish_strengths": [],
            "key_bearish_strengths": [],
            "argument_quality_note": ""
        }

        # Assess bullish arguments
        bull_points = bullish_thesis.get('thesis_points', [])
        bull_confidence = bullish_thesis.get('confidence', 0.0)

        assessment["bullish_strength"] = bull_confidence
        assessment["key_bullish_strengths"] = bull_points[:3]  # Top 3 bullish points

        # Assess bearish arguments
        bear_points = bearish_thesis.get('thesis_points', [])
        bear_confidence = bearish_thesis.get('confidence', 0.0)

        assessment["bearish_strength"] = bear_confidence
        assessment["key_bearish_strengths"] = bear_points[:3]  # Top 3 bearish points

        # Overall argument quality assessment
        strength_diff = abs(bull_confidence - bear_confidence)
        if strength_diff < 0.1:
            assessment["argument_quality_note"] = "Balanced arguments with similar strength on both sides"
        elif strength_diff < 0.3:
            assessment["argument_quality_note"] = "Moderately stronger arguments on one side"
        else:
            assessment["argument_quality_note"] = "Significantly stronger arguments on one side"

        return assessment

    def _determine_final_signal(self, bull_confidence: float, bear_confidence: float,
                             confidence_diff: float, argument_assessment: Dict) -> tuple:
        """Determine final investment signal based on debate analysis"""

        # Consider confidence differential
        if abs(confidence_diff) < 0.1:
            final_signal = "neutral"
            reasoning = "Balanced debate with compelling arguments on both sides"
            final_confidence = max(bull_confidence, bear_confidence)
        elif confidence_diff > 0:
            final_signal = "bullish"
            reasoning = "Bullish arguments more convincing based on strength and confidence"
            final_confidence = bull_confidence
        else:
            final_signal = "bearish"
            reasoning = "Bearish arguments more convincing based on risk assessment"
            final_confidence = bear_confidence

        return final_signal, reasoning, final_confidence

    def _generate_recommendation(self, signal: str, confidence: float,
                               ticker: str, portfolio: Dict) -> Dict[str, Any]:
        """Generate detailed investment recommendation"""
        cash = portfolio.get('cash', 0)
        position = portfolio.get('stock_position', 0)

        recommendation = {
            "action": "hold",  # Default action
            "position_size_guidance": "Maintain current position",
            "risk_level": "Medium",
            "time_horizon": "Medium to Long-term",
            "key_considerations": []
        }

        if signal == "bullish" and confidence > 0.6:
            recommendation["action"] = "buy"
            recommendation["position_size_guidance"] = f"Consider increasing position by up to {int(cash * 0.2):,.0f} units"
            recommendation["risk_level"] = "Low to Medium"
            recommendation["key_considerations"].append("Strong bullish thesis with high confidence")
        elif signal == "bearish" and confidence > 0.6:
            recommendation["action"] = "sell"
            recommendation["position_size_guidance"] = f"Consider reducing position by up to {position * 0.5:.0f} shares"
            recommendation["risk_level"] = "Medium to High"
            recommendation["key_considerations"].append("Significant risk factors identified")
        else:
            recommendation["key_considerations"].append("Balanced outlook suggests patience")

        recommendation["key_considerations"].extend([
            f"Current portfolio: {position} shares, ${cash:,.0f} cash",
            f"Signal confidence: {confidence:.1%}",
            "Monitor for changes in key assumptions and catalysts"
        ])

        return recommendation