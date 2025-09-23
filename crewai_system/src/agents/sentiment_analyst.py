"""
情绪分析师智能体
负责分析市场情绪、新闻情感和社交媒体舆情
"""

import pandas as pd
import numpy as np
import re
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
import math

from crewai_system.src.agents.base_agent import BaseAgent
from crewai_system.src.utils.data_processing import get_data_processor
from crewai_system.src.utils.shared_context import get_global_context
from crewai_system.src.utils.llm_clients import LLMClientFactory


class SentimentAnalyst(BaseAgent):
    """情绪分析师智能体"""

    def __init__(self):
        super().__init__(
            role="市场情绪分析专家",
            goal="分析市场情绪、新闻情感和社交媒体舆情，评估市场情绪状态",
            backstory="""你是一位资深的市场情绪分析师，擅长通过文本分析和情感识别
            来评估市场情绪状态。你能够从新闻、社交媒体、研究报告等多种信息源
            中提取情绪信号，并分析市场参与者的心理状态和预期。
            你的分析为投资决策提供重要的情绪面参考依据。""",
            agent_name="SentimentAnalyst"
        )

        self._data_processor = get_data_processor()
        self._llm_client = None

    @property
    def data_processor(self):
        """获取数据处理器"""
        return getattr(self, '_data_processor', None)

    @property
    def llm_client(self):
        """获取LLM客户端"""
        if self._llm_client is None:
            self._llm_client = LLMClientFactory.create_client()
        return self._llm_client

    def process_task(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理情绪分析任务

        Args:
            task_context: 任务上下文，包含新闻数据等信息

        Returns:
            情绪分析结果
        """
        self.log_execution_start("执行情绪分析")
        print(f"🔍 [DEBUG] ===== SentimentAnalyst 开始处理任务 =====")

        try:
            # 验证输入
            required_fields = ["ticker", "news_data"]
            print(f"🔍 [DEBUG] 验证输入字段，必需字段: {required_fields}")
            print(f"🔍 [DEBUG] task_context keys: {list(task_context.keys())}")

            if not self.validate_input(task_context, required_fields):
                raise ValueError(f"缺少必需字段: {required_fields}")

            ticker = task_context["ticker"]
            news_data = task_context.get("news_data", [])
            # log news_data
            print(f"🔍 [DEBUG] ===== 新闻数据详情 =====")
            print(f"🔍 [DEBUG] 接收到的 news_data 大小: {len(news_data)}")
            print(f"🔍 [DEBUG] news_data 类型: {type(news_data)}")

            if news_data:
                print(f"🔍 [DEBUG] news_data 第一条新闻样本: {json.dumps(news_data[0] if news_data else {}, ensure_ascii=False, indent=2)}")
                # 记录前几条新闻的标题
                for i, news in enumerate(news_data[:3]):
                    title = news.get('title', '无标题')
                    source = news.get('source', '未知来源')
                    print(f"🔍 [DEBUG] 新闻{i+1}: [{source}] {title}")
            else:
                print(f"🔍 [DEBUG] news_data 为空列表!")

            market_data = task_context.get("market_data", {})
            print(f"🔍 [DEBUG] market_data 大小: {len(market_data)}")
            num_of_news = task_context.get("num_of_news", 10)
            show_reasoning = task_context.get("show_reasoning", False)
            print(f"🔍 [DEBUG] 参数 - ticker: {ticker}, num_of_news: {num_of_news}, show_reasoning: {show_reasoning}")

            # 执行情绪分析
            analysis_result = self._perform_sentiment_analysis(
                news_data, market_data, ticker, num_of_news
            )

            # 生成情绪信号
            sentiment_signal = self._generate_sentiment_signal(analysis_result)

            # 记录推理过程
            if show_reasoning:
                reasoning = self._generate_reasoning_report(analysis_result, sentiment_signal)
                self.log_reasoning(reasoning, "情绪分析推理过程")

            result = self.format_agent_output(
                content={
                    "analysis_result": analysis_result,
                    "sentiment_signal": sentiment_signal
                },
                signal=sentiment_signal["direction"],
                confidence=sentiment_signal["confidence"],
                reasoning=sentiment_signal["reasoning"],
                metadata={
                    "ticker": ticker,
                    "analysis_date": datetime.now().isoformat(),
                    "news_count": len(news_data),
                    "sentiment_sources": ["news", "market_data"]
                }
            )
            # log result
            print(f"情绪分析结果: {json.dumps(result, ensure_ascii=False, indent=2)}")

            self.log_execution_complete(f"完成{ticker}的情绪分析")
            return result

        except Exception as e:
            self.log_execution_error(e, "情绪分析执行失败")
            raise

    def _perform_sentiment_analysis(
        self,
        news_data: List[Dict[str, Any]],
        market_data: Dict[str, Any],
        ticker: str,
        num_of_news: int
    ) -> Dict[str, Any]:
        """
        执行情绪分析

        Args:
            news_data: 新闻数据
            market_data: 市场数据
            ticker: 股票代码
            num_of_news: 新闻数量

        Returns:
            情绪分析结果
        """
        print(f"🔍 [DEBUG] ===== 开始执行情绪分析 =====")
        print(f"🔍 [DEBUG] ticker: {ticker}, num_of_news: {num_of_news}")
        print(f"🔍 [DEBUG] 实际新闻数据大小: {len(news_data)}")
        print(f"🔍 [DEBUG] 市场数据 keys: {list(market_data.keys()) if market_data else 'None'}")

        analysis_result = {
            "ticker": ticker,
            "analysis_timestamp": datetime.now().isoformat(),
            "news_sentiment": {},
            "market_sentiment": {},
            "social_sentiment": {},
            "sentiment_trend": {},
            "extreme_sentiment": {},
            "contrarian_signal": {},
            "overall_sentiment": 0
        }

        # 新闻情绪分析
        print(f"🔍 [DEBUG] ===== 开始新闻情绪分析 =====")
        analysis_result["news_sentiment"] = self._analyze_news_sentiment(news_data, num_of_news)
        print(f"🔍 [DEBUG] 新闻情绪分析完成: {analysis_result['news_sentiment'].get('sentiment_score', 0)}")

        # 市场情绪分析
        print(f"🔍 [DEBUG] ===== 开始市场情绪分析 =====")
        analysis_result["market_sentiment"] = self._analyze_market_sentiment(market_data)
        print(f"🔍 [DEBUG] 市场情绪分析完成: {analysis_result['market_sentiment'].get('sentiment_score', 0)}")

        # 社交媒体情绪分析（暂时禁用，需要真实API支持）
        print(f"🔍 [DEBUG] ===== 开始社交媒体情绪分析 =====")
        analysis_result["social_sentiment"] = self._analyze_social_sentiment_basic()
        print(f"🔍 [DEBUG] 社交媒体情绪分析完成")

        # 情绪趋势分析
        print(f"🔍 [DEBUG] ===== 开始情绪趋势分析 =====")
        analysis_result["sentiment_trend"] = self._analyze_sentiment_trend(analysis_result)
        print(f"🔍 [DEBUG] 情绪趋势分析完成")

        # 极端情绪分析
        print(f"🔍 [DEBUG] ===== 开始极端情绪分析 =====")
        analysis_result["extreme_sentiment"] = self._analyze_extreme_sentiment(analysis_result)
        print(f"🔍 [DEBUG] 极端情绪分析完成")

        # 反向信号分析
        print(f"🔍 [DEBUG] ===== 开始反向信号分析 =====")
        analysis_result["contrarian_signal"] = self._analyze_contrarian_signal(analysis_result)
        print(f"🔍 [DEBUG] 反向信号分析完成")

        # 计算综合情绪评分
        print(f"🔍 [DEBUG] ===== 开始计算综合情绪评分 =====")
        analysis_result["overall_sentiment"] = self._calculate_overall_sentiment(analysis_result)
        print(f"🔍 [DEBUG] 综合情绪评分完成: {analysis_result['overall_sentiment']}")

        print(f"🔍 [DEBUG] ===== 情绪分析完成 =====")
        return analysis_result

    def _analyze_news_sentiment(self, news_data: List[Dict[str, Any]], num_of_news: int) -> Dict[str, Any]:
        """使用LLM分析新闻情绪"""
        print(f"🔍 [DEBUG] ===== 开始新闻情绪分析 =====")
        print(f"🔍 [DEBUG] 原始新闻数量: {len(news_data)}, 限制数量: {num_of_news}")

        sentiment_result = {
            "sentiment_score": 0,
            "sentiment_label": "neutral",
            "confidence": 0,
            "news_count": len(news_data),
            "bullish_news": 0,
            "bearish_news": 0,
            "neutral_news": 0,
            "key_topics": [],
            "sentiment_distribution": {},
            "llm_analysis": ""
        }

        if not news_data:
            print(f"🔍 [DEBUG] 新闻数据为空，返回中性情绪结果")
            return sentiment_result

        # 限制新闻数量
        analyzed_news = news_data[:num_of_news]
        print(f"🔍 [DEBUG] 实际分析新闻数量: {len(analyzed_news)}")

        # 使用LLM进行情绪分析
        try:
            print(f"🔍 [DEBUG] ===== 开始LLM新闻情绪分析 =====")
            llm_analysis = self._analyze_news_with_llm(analyzed_news)
            print(f"🔍 [DEBUG] LLM分析成功，更新结果")
            sentiment_result.update(llm_analysis)
        except Exception as e:
            print(f"🔍 [DEBUG] LLM新闻情绪分析失败: {e}")
            print(f"🔍 [DEBUG] ===== 回退到基础关键词分析 =====")
            # 如果LLM分析失败，回退到基础分析
            sentiment_result = self._fallback_news_analysis(analyzed_news, sentiment_result)

        print(f"🔍 [DEBUG] 新闻情绪分析完成: score={sentiment_result.get('sentiment_score', 0)}, label={sentiment_result.get('sentiment_label', 'neutral')}")
        return sentiment_result

    def _analyze_news_with_llm(self, news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """使用LLM分析新闻情绪"""

        # 准备新闻文本
        news_texts = []
        for i, news in enumerate(news_data[:10]):  # 限制分析的新闻数量
            title = news.get("title", "")
            content = news.get("content", "")

            if not isinstance(title, str):
                title = str(title) if title is not None else ""
            if not isinstance(content, str):
                content = str(content) if content is not None else ""

            news_texts.append(f"新闻{i+1}: {title}\n{content[:200]}...\n")

        news_text = "\n\n".join(news_texts)

        # 构建LLM提示
        prompt = f"""
        请分析以下关于股市的新闻内容，评估整体市场情绪。

        新闻内容：
        {news_text}

        请以JSON格式返回分析结果，包含以下字段：
        - sentiment_score: 整体情绪分数（-1到1之间，负数表示悲观，正数表示乐观）
        - sentiment_label: 情绪标签（very_bullish, bullish, neutral, bearish, very_bearish）
        - confidence: 分析置信度（0到1之间）
        - bullish_news: 看涨新闻数量
        - bearish_news: 看跌新闻数量
        - neutral_news: 中性新闻数量
        - key_topics: 关键话题列表
        - reasoning: 分析理由

        请确保返回的是有效的JSON格式。
        """

        # 调用LLM
        try:
            response = self.llm_client.get_completion([
                {"role": "system", "content": "你是一位专业的金融分析师，擅长分析新闻中的市场情绪。请始终以JSON格式返回分析结果。"},
                {"role": "user", "content": prompt}
            ])

            if response:
                # 尝试解析JSON响应
                try:
                    result = json.loads(response)

                    # 计算情绪分布
                    total_news = result.get("bullish_news", 0) + result.get("bearish_news", 0) + result.get("neutral_news", 0)
                    if total_news > 0:
                        sentiment_distribution = {
                            "bullish_ratio": result.get("bullish_news", 0) / total_news,
                            "bearish_ratio": result.get("bearish_news", 0) / total_news,
                            "neutral_ratio": result.get("neutral_news", 0) / total_news
                        }
                    else:
                        sentiment_distribution = {"bullish_ratio": 0, "bearish_ratio": 0, "neutral_ratio": 1}

                    return {
                        "sentiment_score": result.get("sentiment_score", 0),
                        "sentiment_label": result.get("sentiment_label", "neutral"),
                        "confidence": result.get("confidence", 0.5),
                        "bullish_news": result.get("bullish_news", 0),
                        "bearish_news": result.get("bearish_news", 0),
                        "neutral_news": result.get("neutral_news", len(news_data)),
                        "key_topics": result.get("key_topics", []),
                        "sentiment_distribution": sentiment_distribution,
                        "llm_analysis": result.get("reasoning", "")
                    }
                except json.JSONDecodeError:
                    # 尝试提取JSON内容，移除markdown标记
                    import re
                    clean_response = re.sub(r'```json\n?', '', response)
                    clean_response = re.sub(r'\n?```', '', clean_response)
                    try:
                        result = json.loads(clean_response)

                        # 计算情绪分布
                        total_news = result.get("bullish_news", 0) + result.get("bearish_news", 0) + result.get("neutral_news", 0)
                        if total_news > 0:
                            sentiment_distribution = {
                                "bullish_ratio": result.get("bullish_news", 0) / total_news,
                                "bearish_ratio": result.get("bearish_news", 0) / total_news,
                                "neutral_ratio": result.get("neutral_news", 0) / total_news
                            }
                        else:
                            sentiment_distribution = {"bullish_ratio": 0, "bearish_ratio": 0, "neutral_ratio": 1}

                        return {
                            "sentiment_score": result.get("sentiment_score", 0),
                            "sentiment_label": result.get("sentiment_label", "neutral"),
                            "confidence": result.get("confidence", 0.5),
                            "bullish_news": result.get("bullish_news", 0),
                            "bearish_news": result.get("bearish_news", 0),
                            "neutral_news": result.get("neutral_news", len(news_data)),
                            "key_topics": result.get("key_topics", []),
                            "sentiment_distribution": sentiment_distribution,
                            "llm_analysis": result.get("reasoning", "")
                        }
                    except:
                        print(f"LLM返回的不是有效JSON: {response}")
                        raise ValueError("LLM response is not valid JSON")
            else:
                raise ValueError("LLM returned empty response")

        except Exception as e:
            print(f"LLM新闻分析失败: {e}")
            raise

    def _fallback_news_analysis(self, news_data: List[Dict[str, Any]], sentiment_result: Dict[str, Any]) -> Dict[str, Any]:
        """基础新闻情绪分析（作为LLM分析失败的回退方案）"""
        bullish_count = 0
        bearish_count = 0
        neutral_count = len(news_data)

        # 简单的关键词匹配
        bullish_keywords = ["上涨", "增长", "利好", "突破", "反弹", "强势", "盈利", "收入", "利润", "业绩"]
        bearish_keywords = ["下跌", "利空", "跌破", "回调", "弱势", "亏损", "下滑", "下降", "减少", "风险"]

        for news in news_data:
            title = news.get("title", "")
            content = news.get("content", "")

            if not isinstance(title, str):
                title = str(title) if title is not None else ""
            if not isinstance(content, str):
                content = str(content) if content is not None else ""

            text = f"{title} {content}".lower()

            bullish_found = any(keyword in text for keyword in bullish_keywords)
            bearish_found = any(keyword in text for keyword in bearish_keywords)

            if bullish_found and not bearish_found:
                bullish_count += 1
                neutral_count -= 1
            elif bearish_found and not bullish_found:
                bearish_count += 1
                neutral_count -= 1

        # 计算情绪分数
        total = bullish_count + bearish_count + neutral_count
        if total > 0:
            sentiment_score = (bullish_count - bearish_count) / total
        else:
            sentiment_score = 0

        # 确定情绪标签
        if sentiment_score > 0.3:
            sentiment_label = "bullish"
        elif sentiment_score < -0.3:
            sentiment_label = "bearish"
        else:
            sentiment_label = "neutral"

        sentiment_result.update({
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
            "confidence": 0.3,  # 回退分析的置信度较低
            "bullish_news": bullish_count,
            "bearish_news": bearish_count,
            "neutral_news": neutral_count,
            "sentiment_distribution": {
                "bullish_ratio": bullish_count / total if total > 0 else 0,
                "bearish_ratio": bearish_count / total if total > 0 else 0,
                "neutral_ratio": neutral_count / total if total > 0 else 1
            },
            "llm_analysis": "使用基础关键词分析（LLM分析失败）"
        })

        return sentiment_result

    def _analyze_market_sentiment(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析市场情绪"""
        sentiment_result = {
            "sentiment_score": 0,
            "sentiment_label": "neutral",
            "price_change_sentiment": "neutral",
            "volume_sentiment": "neutral",
            "volatility_sentiment": "neutral",
            "technical_sentiment": "neutral"
        }

        try:
            # 价格变化情绪
            price_change = market_data.get("price_change_percent", 0)
            if price_change > 5:
                price_sentiment = "very_bullish"
                price_score = 0.8
            elif price_change > 2:
                price_sentiment = "bullish"
                price_score = 0.4
            elif price_change < -5:
                price_sentiment = "very_bearish"
                price_score = -0.8
            elif price_change < -2:
                price_sentiment = "bearish"
                price_score = -0.4
            else:
                price_sentiment = "neutral"
                price_score = 0

            sentiment_result["price_change_sentiment"] = price_sentiment

            # 成交量情绪
            volume_ratio = market_data.get("volume_ratio", 1.0)
            if volume_ratio > 2.0:
                volume_sentiment = "high_volume_bullish"
                volume_score = 0.3
            elif volume_ratio > 1.5:
                volume_sentiment = "moderate_volume_bullish"
                volume_score = 0.15
            elif volume_ratio < 0.5:
                volume_sentiment = "low_volume_bearish"
                volume_score = -0.3
            else:
                volume_sentiment = "normal_volume"
                volume_score = 0

            sentiment_result["volume_sentiment"] = volume_sentiment

            # 波动率情绪
            volatility = market_data.get("volatility", 0.2)
            if volatility > 0.4:
                volatility_sentiment = "high_volatility_bearish"
                volatility_score = -0.2
            elif volatility < 0.15:
                volatility_sentiment = "low_volatility_bullish"
                volatility_score = 0.2
            else:
                volatility_sentiment = "normal_volatility"
                volatility_score = 0

            sentiment_result["volatility_sentiment"] = volatility_sentiment

            # 技术指标情绪（简化）
            rsi = market_data.get("rsi", 50)
            if rsi > 70:
                technical_sentiment = "overbought_bearish"
                technical_score = -0.3
            elif rsi < 30:
                technical_sentiment = "oversold_bullish"
                technical_score = 0.3
            else:
                technical_sentiment = "neutral"
                technical_score = 0

            sentiment_result["technical_sentiment"] = technical_sentiment

            # 计算综合市场情绪分数
            total_score = price_score + volume_score + volatility_score + technical_score

            if total_score > 0.8:
                sentiment_label = "very_bullish"
            elif total_score > 0.3:
                sentiment_label = "bullish"
            elif total_score < -0.8:
                sentiment_label = "very_bearish"
            elif total_score < -0.3:
                sentiment_label = "bearish"
            else:
                sentiment_label = "neutral"

            sentiment_result["sentiment_score"] = total_score
            sentiment_result["sentiment_label"] = sentiment_label

        except Exception as e:
            print(f"市场情绪分析失败: {e}")
            sentiment_result["error"] = str(e)

        return sentiment_result

    def _analyze_social_sentiment_basic(self) -> Dict[str, Any]:
        """基础社交媒体情绪分析（需要真实API支持）"""
        return {
            "overall_sentiment": 0.0,
            "confidence": 0.0,
            "note": "社交媒体情绪分析需要真实的社交媒体API支持"
        }



    def _analyze_sentiment_trend(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """高级情绪趋势分析"""
        try:
            news_sentiment = analysis_result.get("news_sentiment", {})
            market_sentiment = analysis_result.get("market_sentiment", {})
            social_sentiment = analysis_result.get("social_sentiment", {})

            # 收集各源情绪分数
            sentiment_sources = [
                news_sentiment.get("sentiment_score", 0),
                market_sentiment.get("sentiment_score", 0),
                social_sentiment.get("sentiment_score", 0)
            ]

            # 计算趋势指标
            trend_analysis = self._calculate_sentiment_trend(sentiment_sources)

            # 计算情绪动量
            momentum_analysis = self._calculate_sentiment_momentum(sentiment_sources)

            # 计算情绪波动率
            volatility_analysis = self._calculate_sentiment_volatility(sentiment_sources)

            # 预测情绪趋势
            trend_prediction = self._predict_sentiment_trend(trend_analysis, momentum_analysis)

            return {
                "trend_direction": trend_analysis["direction"],
                "trend_strength": trend_analysis["strength"],
                "trend_stability": trend_analysis["stability"],
                "sentiment_momentum": momentum_analysis["momentum"],
                "momentum_direction": momentum_analysis["direction"],
                "sentiment_volatility": volatility_analysis["volatility"],
                "volatility_trend": volatility_analysis["trend"],
                "trend_prediction": trend_prediction["prediction"],
                "prediction_confidence": trend_prediction["confidence"],
                "time_horizon": trend_prediction["time_horizon"],
                "analysis_period": "short_term"
            }

        except Exception as e:
            print(f"情绪趋势分析失败: {e}")
            return {
                "trend_direction": "stable",
                "trend_strength": 0,
                "sentiment_momentum": 0,
                "error": str(e)
            }

    def _calculate_sentiment_trend(self, sentiment_sources: List[float]) -> Dict[str, Any]:
        """计算情绪趋势"""
        if len(sentiment_sources) < 2:
            return {"direction": "stable", "strength": 0, "stability": 1.0}

        # 简单线性回归计算趋势
        x = list(range(len(sentiment_sources)))
        y = sentiment_sources

        # 计算斜率
        n = len(x)
        x_mean = sum(x) / n
        y_mean = sum(y) / n

        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0

        # 确定趋势方向
        if slope > 0.1:
            direction = "rising"
        elif slope < -0.1:
            direction = "declining"
        else:
            direction = "stable"

        # 计算趋势强度
        strength = min(abs(slope) * 10, 1.0)

        # 计算趋势稳定性（R平方）
        total_variance = sum((y_i - y_mean) ** 2 for y_i in y)
        explained_variance = sum((slope * (x_i - x_mean)) ** 2 for x_i in x)
        r_squared = explained_variance / total_variance if total_variance != 0 else 0

        stability = r_squared

        return {
            "direction": direction,
            "strength": strength,
            "stability": stability,
            "slope": slope
        }

    def _calculate_sentiment_momentum(self, sentiment_sources: List[float]) -> Dict[str, Any]:
        """计算情绪动量"""
        if len(sentiment_sources) < 3:
            return {"momentum": 0, "direction": "neutral"}

        # 计算最近期的变化
        recent_change = sentiment_sources[-1] - sentiment_sources[-2]
        medium_change = sentiment_sources[-2] - sentiment_sources[-3] if len(sentiment_sources) > 2 else 0

        # 计算动量强度
        momentum = (recent_change + medium_change * 0.5) / 1.5

        # 确定动量方向
        if momentum > 0.2:
            direction = "bullish_momentum"
        elif momentum < -0.2:
            direction = "bearish_momentum"
        else:
            direction = "neutral"

        return {
            "momentum": momentum,
            "direction": direction,
            "recent_change": recent_change,
            "medium_change": medium_change
        }

    def _calculate_sentiment_volatility(self, sentiment_sources: List[float]) -> Dict[str, Any]:
        """计算情绪波动率"""
        if len(sentiment_sources) < 2:
            return {"volatility": 0, "trend": "stable"}

        # 计算标准差
        mean_sentiment = sum(sentiment_sources) / len(sentiment_sources)
        variance = sum((s - mean_sentiment) ** 2 for s in sentiment_sources) / len(sentiment_sources)
        volatility = math.sqrt(variance)

        # 确定波动率趋势
        if len(sentiment_sources) >= 4:
            recent_volatility = math.sqrt(sum((s - mean_sentiment) ** 2 for s in sentiment_sources[-2:]) / 2)
            early_volatility = math.sqrt(sum((s - mean_sentiment) ** 2 for s in sentiment_sources[:-2]) / (len(sentiment_sources) - 2))

            if recent_volatility > early_volatility * 1.2:
                trend = "increasing"
            elif recent_volatility < early_volatility * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return {
            "volatility": volatility,
            "trend": trend,
            "mean_sentiment": mean_sentiment
        }

    def _predict_sentiment_trend(self, trend_analysis: Dict[str, Any],
                               momentum_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """预测情绪趋势"""
        prediction = "stable"
        confidence = 0.5
        time_horizon = "short_term"

        # 基于趋势和动量预测
        trend_direction = trend_analysis.get("direction", "stable")
        trend_strength = trend_analysis.get("strength", 0)
        momentum_direction = momentum_analysis.get("direction", "neutral")
        momentum_strength = abs(momentum_analysis.get("momentum", 0))

        # 预测逻辑
        if trend_direction == "rising" and momentum_direction == "bullish_momentum":
            prediction = "continuation_rising"
            confidence = min(0.5 + trend_strength * 0.3 + momentum_strength * 0.2, 0.9)
        elif trend_direction == "declining" and momentum_direction == "bearish_momentum":
            prediction = "continuation_declining"
            confidence = min(0.5 + trend_strength * 0.3 + momentum_strength * 0.2, 0.9)
        elif trend_direction == "rising" and momentum_direction == "bearish_momentum":
            prediction = "potential_reversal"
            confidence = min(0.5 + momentum_strength * 0.4, 0.8)
        elif trend_direction == "declining" and momentum_direction == "bullish_momentum":
            prediction = "potential_reversal"
            confidence = min(0.5 + momentum_strength * 0.4, 0.8)

        # 根据置信度确定时间范围
        if confidence > 0.7:
            time_horizon = "medium_term"
        else:
            time_horizon = "short_term"

        return {
            "prediction": prediction,
            "confidence": confidence,
            "time_horizon": time_horizon
        }

    def _analyze_extreme_sentiment(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """分析极端情绪"""
        extreme_sentiment = {
            "is_extreme": False,
            "extreme_type": None,
            "extreme_level": 0,
            "warning_signals": []
        }

        # 检查新闻情绪是否极端
        news_sentiment = analysis_result.get("news_sentiment", {})
        news_score = news_sentiment.get("sentiment_score", 0)

        if abs(news_score) > 0.6:
            extreme_sentiment["is_extreme"] = True
            extreme_sentiment["extreme_type"] = "extreme_bullish" if news_score > 0 else "extreme_bearish"
            extreme_sentiment["extreme_level"] = abs(news_score)
            extreme_sentiment["warning_signals"].append("新闻情绪极端")

        # 检查市场情绪是否极端
        market_sentiment = analysis_result.get("market_sentiment", {})
        market_score = market_sentiment.get("sentiment_score", 0)

        if abs(market_score) > 1.0:
            extreme_sentiment["is_extreme"] = True
            if not extreme_sentiment["extreme_type"]:
                extreme_sentiment["extreme_type"] = "extreme_bullish" if market_score > 0 else "extreme_bearish"
            extreme_sentiment["extreme_level"] = max(extreme_sentiment["extreme_level"], abs(market_score))
            extreme_sentiment["warning_signals"].append("市场情绪极端")

        return extreme_sentiment

    def _analyze_contrarian_signal(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """分析反向信号"""
        contrarian_signal = {
            "has_contrarian_signal": False,
            "signal_strength": 0,
            "signal_direction": None,
            "rationale": []
        }

        extreme_sentiment = analysis_result.get("extreme_sentiment", {})

        # 极端情绪时产生反向信号
        if extreme_sentiment.get("is_extreme"):
            extreme_type = extreme_sentiment.get("extreme_type")
            extreme_level = extreme_sentiment.get("extreme_level", 0)

            if extreme_level > 0.6:
                contrarian_signal["has_contrarian_signal"] = True
                contrarian_signal["signal_strength"] = extreme_level

                if extreme_type == "extreme_bullish":
                    contrarian_signal["signal_direction"] = "bearish_reversal"
                    contrarian_signal["rationale"].append("极度乐观，谨防回调")
                elif extreme_type == "extreme_bearish":
                    contrarian_signal["signal_direction"] = "bullish_reversal"
                    contrarian_signal["rationale"].append("极度悲观，可能存在反弹机会")

        return contrarian_signal

    def _calculate_overall_sentiment(self, analysis_result: Dict[str, Any]) -> float:
        """高级综合情绪评分计算"""
        try:
            # 获取各源情绪数据
            news_sentiment = analysis_result.get("news_sentiment", {})
            market_sentiment = analysis_result.get("market_sentiment", {})
            social_sentiment = analysis_result.get("social_sentiment", {})
            sentiment_trend = analysis_result.get("sentiment_trend", {})
            extreme_sentiment = analysis_result.get("extreme_sentiment", {})

            # 提取情绪分数
            news_score = news_sentiment.get("sentiment_score", 0)
            market_score = market_sentiment.get("sentiment_score", 0)
            social_score = social_sentiment.get("sentiment_score", 0)

            # 提取置信度
            news_confidence = news_sentiment.get("confidence", 0)
            market_confidence = 0.8  # 市场数据置信度通常较高
            social_confidence = social_sentiment.get("confidence", 0)

            # 提取趋势信息
            trend_strength = sentiment_trend.get("trend_strength", 0)
            trend_momentum = sentiment_trend.get("sentiment_momentum", 0)

            # 提取极端情绪信息
            extreme_level = extreme_sentiment.get("extreme_level", 0)
            is_extreme = extreme_sentiment.get("is_extreme", False)

            # 动态权重计算
            dynamic_weights = self._calculate_dynamic_sentiment_weights(
                news_score, market_score, social_score,
                news_confidence, market_confidence, social_confidence,
                trend_strength, extreme_level
            )

            # 基础情绪计算
            base_scores = [news_score, market_score, social_score]
            base_weights = dynamic_weights["base_weights"]
            base_sentiment = sum(score * weight for score, weight in zip(base_scores, base_weights))

            # 趋势调整
            trend_adjustment = trend_strength * trend_momentum * 0.1
            trend_adjusted_sentiment = base_sentiment + trend_adjustment

            # 极端情绪调整
            extreme_adjustment = 0
            if is_extreme:
                extreme_adjustment = extreme_level * 0.2 * (1 if base_sentiment > 0 else -1)

            # 时间衰减调整（较新的数据权重更高）
            time_decay_factors = self._calculate_time_decay_factors(analysis_result)
            time_adjusted_sentiment = trend_adjusted_sentiment * time_decay_factors

            # 情绪平滑处理
            smoothed_sentiment = self._apply_sentiment_smoothing(time_adjusted_sentiment, base_scores)

            # 极值处理
            final_sentiment = self._handle_sentiment_extremes(smoothed_sentiment, extreme_sentiment)

            return max(-1, min(1, final_sentiment))

        except Exception as e:
            print(f"综合情绪计算失败: {e}")
            return 0

    def _calculate_dynamic_sentiment_weights(self, news_score: float, market_score: float, social_score: float,
                                          news_confidence: float, market_confidence: float, social_confidence: float,
                                          trend_strength: float, extreme_level: float) -> Dict[str, Any]:
        """动态计算情绪权重"""
        # 基础权重
        base_news_weight = 0.4
        base_market_weight = 0.35
        base_social_weight = 0.25

        # 根据置信度调整权重
        confidence_adjustment = 0.3
        news_weight = base_news_weight + (news_confidence - 0.5) * confidence_adjustment
        market_weight = base_market_weight + (market_confidence - 0.5) * confidence_adjustment * 0.5
        social_weight = base_social_weight + (social_confidence - 0.5) * confidence_adjustment * 0.5

        # 根据情绪一致性调整权重
        sentiment_consistency = self._calculate_sentiment_consistency([news_score, market_score, social_score])
        consistency_bonus = sentiment_consistency * 0.2

        if sentiment_consistency > 0.7:
            # 高一致性时，强化占主导情绪的权重
            max_sentiment = max(abs(news_score), abs(market_score), abs(social_score))
            if abs(news_score) == max_sentiment:
                news_weight += consistency_bonus
            elif abs(market_score) == max_sentiment:
                market_weight += consistency_bonus
            else:
                social_weight += consistency_bonus

        # 根据趋势强度调整
        trend_adjustment = trend_strength * 0.1
        if trend_strength > 0.5:
            # 强趋势时，增加市场情绪权重
            market_weight += trend_adjustment

        # 根据极端情绪调整
        extreme_adjustment = extreme_level * 0.15
        if extreme_level > 0.5:
            # 极端情绪时，增加新闻情绪权重
            news_weight += extreme_adjustment

        # 归一化权重
        total_weight = news_weight + market_weight + social_weight
        normalized_weights = [
            news_weight / total_weight,
            market_weight / total_weight,
            social_weight / total_weight
        ]

        return {
            "base_weights": normalized_weights,
            "raw_weights": [news_weight, market_weight, social_weight],
            "consistency_score": sentiment_consistency,
            "adjustment_factors": {
                "confidence": confidence_adjustment,
                "consistency": consistency_bonus,
                "trend": trend_adjustment,
                "extreme": extreme_adjustment
            }
        }

    def _calculate_sentiment_consistency(self, sentiment_scores: List[float]) -> float:
        """计算情绪一致性"""
        if len(sentiment_scores) < 2:
            return 0.5

        # 计算符号一致性
        signs = [1 if s > 0 else -1 if s < 0 else 0 for s in sentiment_scores]
        sign_consistency = abs(sum(signs)) / len(signs) if signs else 0

        # 计算数值一致性（变异系数的倒数）
        mean_score = np.mean(sentiment_scores)
        std_score = np.std(sentiment_scores)
        cv = std_score / abs(mean_score) if mean_score != 0 else 1
        numerical_consistency = 1 / (1 + cv)

        # 综合一致性
        consistency = (sign_consistency * 0.6 + numerical_consistency * 0.4)

        return consistency

    def _calculate_time_decay_factors(self, analysis_result: Dict[str, Any]) -> float:
        """计算时间衰减因子"""
        # 简化的时间衰减计算
        # 在实际实现中，需要根据各数据源的时间戳进行精确计算
        return 1.0  # 目前返回无衰减

    def _apply_sentiment_smoothing(self, sentiment: float, historical_scores: List[float]) -> float:
        """应用情绪平滑处理"""
        if not historical_scores:
            return sentiment

        # 指数移动平均
        alpha = 0.3  # 平滑因子
        historical_avg = np.mean(historical_scores)
        smoothed_sentiment = alpha * sentiment + (1 - alpha) * historical_avg

        return smoothed_sentiment

    def _handle_sentiment_extremes(self, sentiment: float, extreme_sentiment: Dict[str, Any]) -> float:
        """处理情绪极值"""
        if extreme_sentiment.get("is_extreme"):
            extreme_level = extreme_sentiment.get("extreme_level", 0)
            extreme_type = extreme_sentiment.get("extreme_type")

            # 极端情绪时进行适度调整，避免过度反应
            if extreme_level > 0.7:
                adjustment_factor = 0.9  # 10%的衰减
                sentiment *= adjustment_factor

        return sentiment

    def _extract_key_topics(self, news_data: List[Dict[str, Any]]) -> List[str]:
        """提取关键话题"""
        topics = set()

        # 定义关键话题关键词
        topic_keywords = {
            "盈利": ["盈利", "利润", "业绩", "收入", "收益"],
            "增长": ["增长", "扩张", "发展", "提升"],
            "政策": ["政策", "监管", "法规", "政府"],
            "市场": ["市场", "行业", "竞争", "份额"],
            "技术": ["技术", "创新", "研发", "科技"],
            "风险": ["风险", "危机", "问题", "挑战"],
            "并购": ["并购", "收购", "重组", "合并"]
        }

        for news in news_data:
            title = news.get("title", "")
            content = news.get("content", "")

            # 确保标题和内容都是字符串
            if not isinstance(title, str):
                title = str(title) if title is not None else ""
            if not isinstance(content, str):
                content = str(content) if content is not None else ""

            full_text = f"{title} {content}".lower()

            for topic, keywords in topic_keywords.items():
                for keyword in keywords:
                    if keyword in full_text:
                        topics.add(topic)
                        break

        return list(topics)

    def _generate_sentiment_signal(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成情绪信号"""
        overall_sentiment = analysis_result.get("overall_sentiment", 0)
        contrarian_signal = analysis_result.get("contrarian_signal", {})
        extreme_sentiment = analysis_result.get("extreme_sentiment", {})

        # 基础情绪信号
        if overall_sentiment > 0.4:
            base_direction = "bullish"
            base_confidence = min(overall_sentiment * 80, 90)
        elif overall_sentiment < -0.4:
            base_direction = "bearish"
            base_confidence = min(abs(overall_sentiment) * 80, 90)
        else:
            base_direction = "neutral"
            base_confidence = 50

        # 检查反向信号
        if contrarian_signal.get("has_contrarian_signal"):
            signal_direction = contrarian_signal.get("signal_direction")
            signal_strength = contrarian_signal.get("signal_strength", 0)

            # 反向信号可能改变最终建议
            if signal_strength > 0.7:
                if signal_direction == "bullish_reversal":
                    base_direction = "bullish"
                    base_confidence = max(base_confidence, signal_strength * 70)
                elif signal_direction == "bearish_reversal":
                    base_direction = "bearish"
                    base_confidence = max(base_confidence, signal_strength * 70)

        # 生成推理
        reasoning_parts = []

        if base_direction == "bullish":
            reasoning_parts.append("市场情绪偏向乐观")
        elif base_direction == "bearish":
            reasoning_parts.append("市场情绪偏向悲观")
        else:
            reasoning_parts.append("市场情绪相对中性")

        if contrarian_signal.get("has_contrarian_signal"):
            reasoning_parts.extend(contrarian_signal.get("rationale", []))

        if extreme_sentiment.get("is_extreme"):
            reasoning_parts.append("检测到极端情绪信号")

        reasoning = "；".join(reasoning_parts) if reasoning_parts else "情绪信号不明确"

        return {
            "direction": base_direction,
            "confidence": base_confidence,
            "reasoning": reasoning,
            "overall_sentiment": overall_sentiment,
            "has_contrarian_signal": contrarian_signal.get("has_contrarian_signal", False),
            "is_extreme_sentiment": extreme_sentiment.get("is_extreme", False)
        }

    def _generate_reasoning_report(self, analysis_result: Dict[str, Any],
                                 sentiment_signal: Dict[str, Any]) -> str:
        """生成推理报告"""
        report = []

        # 新闻情绪
        news_sentiment = analysis_result.get("news_sentiment", {})
        news_score = news_sentiment.get("sentiment_score", 0)
        news_label = news_sentiment.get("sentiment_label", "neutral")
        news_count = news_sentiment.get("news_count", 0)

        report.append(f"新闻情绪: {news_label} (评分: {news_score:.2f})")
        report.append(f"分析新闻数量: {news_count}")

        # 情绪分布
        sentiment_dist = news_sentiment.get("sentiment_distribution", {})
        bullish_ratio = sentiment_dist.get("bullish_ratio", 0) * 100
        bearish_ratio = sentiment_dist.get("bearish_ratio", 0) * 100
        neutral_ratio = sentiment_dist.get("neutral_ratio", 0) * 100

        report.append(f"情绪分布: 看涨{bullish_ratio:.1f}%, 看跌{bearish_ratio:.1f}%, 中性{neutral_ratio:.1f}%")

        # 市场情绪
        market_sentiment = analysis_result.get("market_sentiment", {})
        market_label = market_sentiment.get("sentiment_label", "neutral")
        market_score = market_sentiment.get("sentiment_score", 0)

        report.append(f"市场情绪: {market_label} (评分: {market_score:.2f})")

        # 综合情绪
        overall_sentiment = analysis_result.get("overall_sentiment", 0)
        report.append(f"综合情绪评分: {overall_sentiment:.2f}")

        # 极端情绪
        extreme_sentiment = analysis_result.get("extreme_sentiment", {})
        if extreme_sentiment.get("is_extreme"):
            extreme_type = extreme_sentiment.get("extreme_type")
            report.append(f"检测到极端情绪: {extreme_type}")

        # 反向信号
        contrarian_signal = analysis_result.get("contrarian_signal", {})
        if contrarian_signal.get("has_contrarian_signal"):
            signal_direction = contrarian_signal.get("signal_direction")
            report.append(f"反向信号: {signal_direction}")

        # 最终信号
        direction = sentiment_signal.get("direction", "neutral")
        confidence = sentiment_signal.get("confidence", 50)
        reasoning = sentiment_signal.get("reasoning", "")

        report.append(f"最终情绪信号: {direction}")
        report.append(f"信号置信度: {confidence:.1f}%")
        report.append(f"主要理由: {reasoning}")

        return "\n".join(report)

    def get_required_fields(self) -> List[str]:
        """获取任务必需字段"""
        return ["ticker", "news_data"]