"""
技术分析师智能体
负责分析价格趋势、技术指标和交易信号
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from agents.base_agent import BaseAgent
from utils.data_processing import get_data_processor
from utils.shared_context import get_global_context


class TechnicalAnalyst(BaseAgent):
    """技术分析师智能体"""

    def __init__(self):
        super().__init__(
            role="技术分析专家",
            goal="分析股票价格趋势和技术指标，提供技术面投资建议",
            backstory="""你是一位资深的技术分析师，精通各种技术分析方法和指标。
            你擅长识别价格趋势、支撑阻力位、图表形态，并能运用多种技术指标
            （如移动平均线、RSI、MACD、布林带等）进行综合分析。
            你的分析将为投资决策提供重要的技术面参考依据。""",
            agent_name="TechnicalAnalyst"
        )

        self._data_processor = get_data_processor()

    @property
    def data_processor(self):
        """获取数据处理器"""
        return getattr(self, '_data_processor', None)

    def process_task(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理技术分析任务

        Args:
            task_context: 任务上下文，包含价格数据等信息

        Returns:
            技术分析结果
        """
        self.log_execution_start("执行技术分析")

        try:
            # 验证输入
            required_fields = ["ticker"]
            if not self.validate_input(task_context, required_fields):
                raise ValueError(f"缺少必需字段: {required_fields}")

            ticker = task_context["ticker"]
            show_reasoning = task_context.get("show_reasoning", False)

            # 导入必要的模块
            import numpy as np
            import pandas as pd
            from datetime import datetime, timedelta

            # 从数据流管理器获取价格数据
            from utils.data_flow_manager import data_flow_manager
            prices_data = data_flow_manager.retrieve_data("market_data", ticker)

            self.logger.info(f"从数据流管理器获取价格数据: {type(prices_data)}")

            if prices_data is None:
                self.logger.warning("无法从数据流管理器获取价格数据，尝试使用上下文数据")
                # 如果数据流管理器没有数据，尝试从上下文获取
                if "prices" in task_context:
                    prices_data = task_context["prices"]
                else:
                    # 无法获取价格数据，抛出错误
                    self.logger.error(f"无法获取{ticker}的价格数据，技术分析无法进行")
                    raise ValueError(f"缺少{ticker}的价格数据，无法进行技术分析")

            self.logger.info(f"价格数据样本: {str(prices_data)[:300]}...")

            # 确保价格数据是列表格式
            if not isinstance(prices_data, list):
                if isinstance(prices_data, dict):
                    # 检查多个可能的嵌套结构
                    if "prices" in prices_data:
                        prices_data = prices_data["prices"]
                    elif "content" in prices_data and isinstance(prices_data["content"], dict) and "prices" in prices_data["content"]:
                        prices_data = prices_data["content"]["prices"]
                    elif "content" in prices_data and isinstance(prices_data["content"], list):
                        prices_data = prices_data["content"]
                    else:
                        raise ValueError(f"价格数据格式不正确，期望列表，实际: {type(prices_data)}")
                else:
                    raise ValueError(f"价格数据格式不正确，期望列表，实际: {type(prices_data)}")

            prices_df = self.data_processor.dict_to_pandas(prices_data)

            if prices_df.empty:
                raise ValueError("价格数据为空，无法进行技术分析")

            # 执行技术分析
            try:
                self.logger.info(f"开始执行技术分析，DataFrame形状: {prices_df.shape}")
                self.logger.info(f"DataFrame列: {list(prices_df.columns)}")
                analysis_result = self._perform_technical_analysis(prices_df, ticker)
                self.logger.info("技术分析执行完成")
            except Exception as e:
                self.logger.error(f"技术分析执行失败: {e}")
                raise

            # 生成交易信号
            try:
                trading_signal = self._generate_trading_signal(analysis_result)
                self.logger.info(f"交易信号生成完成: {trading_signal['direction']}")
            except Exception as e:
                self.logger.error(f"交易信号生成失败: {e}")
                raise

            # 记录推理过程
            if show_reasoning:
                reasoning = self._generate_reasoning_report(analysis_result, trading_signal)
                self.log_reasoning(reasoning, "技术分析推理过程")

            result = self.format_agent_output(
                content={
                    "analysis_result": analysis_result,
                    "trading_signal": trading_signal
                },
                signal=trading_signal["direction"],
                confidence=trading_signal["confidence"],
                reasoning=trading_signal["reasoning"],
                metadata={
                    "ticker": ticker,
                    "analysis_date": datetime.now().isoformat(),
                    "data_points": len(prices_df)
                }
            )

            self.log_execution_complete(f"完成{ticker}的技术分析")
            return result

        except Exception as e:
            self.log_execution_error(e, "技术分析执行失败")
            raise

    def _perform_technical_analysis(self, prices_df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """
        执行技术分析

        Args:
            prices_df: 价格数据DataFrame
            ticker: 股票代码

        Returns:
            技术分析结果
        """
        self.logger.info(f"开始对{ticker}进行技术分析，数据点数: {len(prices_df)}")

        analysis_result = {
            "ticker": ticker,
            "analysis_timestamp": datetime.now().isoformat(),
            "indicators": {},
            "patterns": {},
            "trends": {},
            "support_resistance": {},
            "momentum": {},
            "volatility": {}
        }

        # 计算技术指标
        analysis_result["indicators"] = self._calculate_all_indicators(prices_df)

        # 分析趋势
        analysis_result["trends"] = self._analyze_trends(prices_df, analysis_result["indicators"])

        # 识别图表形态
        analysis_result["patterns"] = self._identify_patterns(prices_df)

        # 计算支撑阻力位
        analysis_result["support_resistance"] = self._calculate_support_resistance(prices_df)

        # 分析动量指标
        analysis_result["momentum"] = self._analyze_momentum(analysis_result["indicators"])

        # 分析波动性
        analysis_result["volatility"] = self._analyze_volatility(prices_df)

        return analysis_result

    def _calculate_all_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        计算所有技术指标

        Args:
            df: 价格数据DataFrame

        Returns:
            技术指标字典
        """
        indicators = {}

        # 确保数据包含必要的列
        required_columns = ['close', 'high', 'low', 'volume']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0

        # 移动平均线
        indicators["moving_averages"] = self._calculate_moving_averages(df)

        # 指数移动平均线
        indicators["ema"] = self._calculate_ema_indicators(df)

        # RSI
        indicators["rsi"] = self._calculate_rsi(df)

        # 多周期RSI
        indicators["multi_rsi"] = self._calculate_multi_rsi(df)

        # MACD
        indicators["macd"] = self._calculate_macd(df)

        # 布林带
        indicators["bollinger_bands"] = self._calculate_bollinger_bands(df)

        # 威廉指标
        indicators["williams_r"] = self._calculate_williams_r(df)

        # 顺势指标
        indicators["cci"] = self._calculate_cci(df)

        # KDJ
        indicators["kdj"] = self._calculate_kdj(df)

        # 布林带宽度指标
        indicators["bwi"] = self._calculate_bwi(df)

        # 成交量指标
        indicators["volume_indicators"] = self._calculate_volume_indicators(df)

        # 能量潮指标
        indicators["obv"] = self._calculate_obv(df)

        # 资金流量指标
        indicators["mfi"] = self._calculate_mfi(df)

        # 成交量价格趋势
        indicators["vpt"] = self._calculate_vpt(df)

        # 高级技术指标
        indicators["advanced_indicators"] = self._calculate_advanced_indicators(df)

        # 波动率指标
        indicators["volatility_indicators"] = self._calculate_volatility_indicators(df)

        # 市场广度指标
        indicators["breadth_indicators"] = self._calculate_breadth_indicators(df)

        # 量化指标
        indicators["quantitative_indicators"] = self._calculate_quantitative_indicators(df)

        return indicators

    def _calculate_moving_averages(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算移动平均线"""
        ma_result = {}

        periods = [5, 10, 20, 30, 60]
        for period in periods:
            ma_key = f"ma_{period}"
            ma_result[ma_key] = df['close'].rolling(window=period).mean().iloc[-1]

        # 计算移动平均线排列
        ma_5 = ma_result.get("ma_5", 0)
        ma_10 = ma_result.get("ma_10", 0)
        ma_20 = ma_result.get("ma_20", 0)
        ma_30 = ma_result.get("ma_30", 0)

        if ma_5 > ma_10 > ma_20 > ma_30:
            ma_arrangement = "bullish_alignment"  # 多头排列
        elif ma_5 < ma_10 < ma_20 < ma_30:
            ma_arrangement = "bearish_alignment"  # 空头排列
        else:
            ma_arrangement = "mixed"  # 混合排列

        ma_result["arrangement"] = ma_arrangement

        return ma_result

    def _calculate_ema_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算指数移动平均线指标"""
        ema_result = {}
        periods = [5, 10, 20, 50]
        closes = df['close'].values

        for period in periods:
            if len(closes) >= period:
                ema_result[f"ema_{period}"] = self._calculate_ema(closes, period)

        # 计算EMA排列
        if len(ema_result) >= 3:
            ema_5 = ema_result.get("ema_5", 0)
            ema_10 = ema_result.get("ema_10", 0)
            ema_20 = ema_result.get("ema_20", 0)

            if ema_5 > ema_10 > ema_20:
                ema_result["arrangement"] = "bullish_alignment"
            elif ema_5 < ema_10 < ema_20:
                ema_result["arrangement"] = "bearish_alignment"
            else:
                ema_result["arrangement"] = "mixed"

        return ema_result

    def _calculate_multi_rsi(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算多周期RSI指标"""
        periods = [7, 14, 21]
        multi_rsi = {}

        for period in periods:
            rsi_data = self._calculate_rsi(df, period)
            multi_rsi[f"rsi_{period}"] = rsi_data

        # 综合RSI信号
        rsi_signals = []
        for period, rsi_data in multi_rsi.items():
            if rsi_data.get("signal") == "oversold":
                rsi_signals.append("bullish")
            elif rsi_data.get("signal") == "overbought":
                rsi_signals.append("bearish")

        if rsi_signals.count("bullish") >= 2:
            multi_rsi["overall_signal"] = "strong_bullish"
        elif rsi_signals.count("bearish") >= 2:
            multi_rsi["overall_signal"] = "strong_bearish"
        else:
            multi_rsi["overall_signal"] = "mixed"

        return multi_rsi

    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
        """计算威廉指标"""
        if len(df) < period:
            return {"williams_r": -50, "signal": "neutral"}

        high_period = df['high'].rolling(window=period).max()
        low_period = df['low'].rolling(window=period).min()

        williams_r = -100 * (high_period - df['close']) / (high_period - low_period)
        current_wr = williams_r.iloc[-1]

        # 威廉指标信号
        if current_wr <= -80:
            signal = "oversold"
        elif current_wr >= -20:
            signal = "overbought"
        else:
            signal = "neutral"

        return {
            "williams_r": current_wr,
            "signal": signal,
            "period": period
        }

    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> Dict[str, Any]:
        """计算顺势指标"""
        if len(df) < period:
            return {"cci": 0, "signal": "neutral"}

        typical_price = (df['high'] + df['low'] + df['close']) / 3
        cci = (typical_price - typical_price.rolling(window=period).mean()) / (0.015 * typical_price.rolling(window=period).std())
        current_cci = cci.iloc[-1]

        # CCI信号
        if current_cci > 100:
            signal = "overbought"
        elif current_cci < -100:
            signal = "oversold"
        else:
            signal = "neutral"

        return {
            "cci": current_cci,
            "signal": signal,
            "period": period
        }

    def _calculate_bwi(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算布林带宽度指标"""
        bb_data = self._calculate_bollinger_bands(df)
        upper = bb_data.get("upper", 0)
        lower = bb_data.get("lower", 0)
        middle = bb_data.get("middle", 0)

        if middle > 0:
            bwi = (upper - lower) / middle
        else:
            bwi = 0

        # BWI信号
        if bwi > 0.1:
            signal = "high_volatility"
        elif bwi < 0.05:
            signal = "low_volatility"
        else:
            signal = "normal_volatility"

        return {
            "bwi": bwi,
            "signal": signal
        }

    def _calculate_obv(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算能量潮指标"""
        if len(df) < 2:
            return {"obv": 0, "trend": "neutral"}

        obv = np.zeros(len(df))
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv[i] = obv[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv[i] = obv[i-1] - df['volume'].iloc[i]
            else:
                obv[i] = obv[i-1]

        current_obv = obv[-1]
        obv_ma = np.mean(obv[-20:]) if len(obv) >= 20 else current_obv

        # OBV趋势
        if current_obv > obv_ma * 1.05:
            obv_trend = "bullish"
        elif current_obv < obv_ma * 0.95:
            obv_trend = "bearish"
        else:
            obv_trend = "neutral"

        return {
            "obv": current_obv,
            "obv_ma": obv_ma,
            "trend": obv_trend
        }

    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
        """计算资金流量指标"""
        if len(df) < period + 1:
            return {"mfi": 50, "signal": "neutral"}

        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']

        positive_flow = np.where(np.diff(typical_price) > 0, money_flow[1:], 0)
        negative_flow = np.where(np.diff(typical_price) < 0, money_flow[1:], 0)

        positive_mf = np.sum(positive_flow[-period:])
        negative_mf = np.sum(negative_flow[-period:])

        if negative_mf == 0:
            mfi = 100
        else:
            mfi = 100 - (100 / (1 + positive_mf / negative_mf))

        # MFI信号
        if mfi > 80:
            signal = "overbought"
        elif mfi < 20:
            signal = "oversold"
        else:
            signal = "neutral"

        return {
            "mfi": mfi,
            "signal": signal,
            "period": period
        }

    def _calculate_vpt(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算成交量价格趋势指标"""
        if len(df) < 2:
            return {"vpt": 0, "trend": "neutral"}

        vpt = np.zeros(len(df))
        for i in range(1, len(df)):
            price_change = (df['close'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
            vpt[i] = vpt[i-1] + df['volume'].iloc[i] * price_change

        current_vpt = vpt[-1]
        vpt_ma = np.mean(vpt[-20:]) if len(vpt) >= 20 else current_vpt

        # VPT趋势
        if current_vpt > vpt_ma * 1.05:
            vpt_trend = "bullish"
        elif current_vpt < vpt_ma * 0.95:
            vpt_trend = "bearish"
        else:
            vpt_trend = "neutral"

        return {
            "vpt": current_vpt,
            "vpt_ma": vpt_ma,
            "trend": vpt_trend
        }

    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
        """计算RSI指标"""
        closes = df['close'].values
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        if len(closes) < period + 1:
            return {"rsi": 50, "signal": "neutral"}

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        # RSI信号
        if rsi > 70:
            signal = "overbought"
        elif rsi < 30:
            signal = "oversold"
        else:
            signal = "neutral"

        return {
            "rsi": rsi,
            "signal": signal,
            "period": period
        }

    def _calculate_macd(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算MACD指标"""
        closes = df['close'].values

        if len(closes) < 26:
            return {"macd": 0, "signal": 0, "histogram": 0, "signal_type": "neutral"}

        # 计算EMA
        ema_12 = self._calculate_ema(closes, 12)
        ema_26 = self._calculate_ema(closes, 26)

        macd_line = ema_12 - ema_26
        signal_line = self._calculate_ema(np.append(np.full(8, np.nan), macd_line), 9)
        histogram = macd_line - signal_line

        # MACD信号
        if histogram > 0 and macd_line > signal_line:
            signal_type = "bullish"
        elif histogram < 0 and macd_line < signal_line:
            signal_type = "bearish"
        else:
            signal_type = "neutral"

        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram,
            "signal_type": signal_type
        }

    def _calculate_ema(self, data: np.ndarray, period: int) -> float:
        """计算指数移动平均线"""
        if len(data) < period:
            return data[-1] if len(data) > 0 else 0

        alpha = 2 / (period + 1)
        ema = data[0]
        for price in data[1:]:
            ema = alpha * price + (1 - alpha) * ema

        return ema

    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> Dict[str, Any]:
        """计算布林带"""
        if len(df) < period:
            return {"upper": 0, "middle": 0, "lower": 0, "position": "unknown"}

        closes = df['close'].values
        middle = np.mean(closes[-period:])
        std = np.std(closes[-period:])

        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)

        current_price = closes[-1]

        # 价格在布林带中的位置
        if current_price > upper:
            position = "above_upper"  # 超买
        elif current_price < lower:
            position = "below_lower"  # 超卖
        else:
            position = "within_bands"  # 正常

        return {
            "upper": upper,
            "middle": middle,
            "lower": lower,
            "position": position,
            "bandwidth": (upper - lower) / middle if middle > 0 else 0
        }

    def _calculate_kdj(self, df: pd.DataFrame, period: int = 9) -> Dict[str, Any]:
        """计算KDJ指标"""
        if len(df) < period:
            return {"k": 50, "d": 50, "j": 50, "signal": "neutral"}

        high_9 = df['high'].rolling(window=period).max()
        low_9 = df['low'].rolling(window=period).min()

        rsv = (df['close'] - low_9) / (high_9 - low_9) * 100

        k = rsv.ewm(com=2).mean()
        d = k.ewm(com=2).mean()
        j = 3 * k - 2 * d

        current_k = k.iloc[-1]
        current_d = d.iloc[-1]
        current_j = j.iloc[-1]

        # KDJ信号
        if current_k > 80 and current_d > 80:
            signal = "overbought"
        elif current_k < 20 and current_d < 20:
            signal = "oversold"
        else:
            signal = "neutral"

        return {
            "k": current_k,
            "d": current_d,
            "j": current_j,
            "signal": signal
        }

    def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算成交量指标"""
        if len(df) < 20:
            return {"volume_ma": 0, "volume_ratio": 1.0, "volume_signal": "normal"}

        # 成交量移动平均
        volume_ma = df['volume'].rolling(window=20).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]

        # 成交量比率
        volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1.0

        # 成交量信号
        if volume_ratio > 2.0:
            volume_signal = "high_volume"
        elif volume_ratio < 0.5:
            volume_signal = "low_volume"
        else:
            volume_signal = "normal"

        return {
            "volume_ma": volume_ma,
            "volume_ratio": volume_ratio,
            "volume_signal": volume_signal,
            "current_volume": current_volume
        }

    def _analyze_trends(self, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """分析价格趋势"""
        if len(df) < 10:
            return {"trend": "unknown", "strength": 0}

        closes = df['close'].values

        # 计算不同周期的趋势
        short_trend = self._calculate_trend_direction(closes[-5:])  # 5日趋势
        medium_trend = self._calculate_trend_direction(closes[-10:])  # 10日趋势
        long_trend = self._calculate_trend_direction(closes[-20:])  # 20日趋势

        # 综合趋势判断
        trend_signals = [short_trend, medium_trend, long_trend]
        bullish_count = trend_signals.count("bullish")
        bearish_count = trend_signals.count("bearish")

        if bullish_count >= 2:
            overall_trend = "bullish"
        elif bearish_count >= 2:
            overall_trend = "bearish"
        else:
            overall_trend = "sideways"

        # 趋势强度
        price_change = (closes[-1] - closes[0]) / closes[0] * 100
        strength = min(abs(price_change) / 2, 100)  # 0-100的强度值

        return {
            "overall_trend": overall_trend,
            "short_trend": short_trend,
            "medium_trend": medium_trend,
            "long_trend": long_trend,
            "strength": strength,
            "price_change_percent": price_change
        }

    def _calculate_trend_direction(self, prices: np.ndarray) -> str:
        """计算趋势方向"""
        if len(prices) < 2:
            return "unknown"

        # 使用线性回归判断趋势
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]

        if slope > 0:
            return "bullish"
        elif slope < 0:
            return "bearish"
        else:
            return "sideways"

    def _identify_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """识别图表形态（简化版本）"""
        patterns = {
            "detected_patterns": [],
            "pattern_signals": []
        }

        # 简化的形态识别
        closes = df['close'].values
        if len(closes) < 5:
            return patterns

        # 检测连续上涨/下跌
        recent_changes = np.diff(closes[-5:])
        consecutive_up = sum(recent_changes > 0) >= 4
        consecutive_down = sum(recent_changes < 0) >= 4

        if consecutive_up:
            patterns["detected_patterns"].append("consecutive_rise")
            patterns["pattern_signals"].append("bullish_continuation")
        elif consecutive_down:
            patterns["detected_patterns"].append("consecutive_fall")
            patterns["pattern_signals"].append("bearish_continuation")

        return patterns

    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算支撑阻力位（简化版本）"""
        if len(df) < 20:
            return {"support": 0, "resistance": 0}

        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values

        current_price = closes[-1]

        # 简单计算：最近20个交易日的最高价和最低价
        resistance = np.max(highs[-20:])
        support = np.min(lows[-20:])

        # 价格位置
        if current_price > resistance * 0.95:
            price_position = "near_resistance"
        elif current_price < support * 1.05:
            price_position = "near_support"
        else:
            price_position = "middle_range"

        return {
            "support": support,
            "resistance": resistance,
            "current_price": current_price,
            "price_position": price_position,
            "support_resistance_range": resistance - support
        }

    def _analyze_momentum(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """分析动量指标"""
        momentum_signals = []
        signal_strength = {"bullish": 0, "bearish": 0}

        # RSI动量
        rsi_data = indicators.get("rsi", {})
        if rsi_data.get("signal") == "oversold":
            momentum_signals.append("bullish_reversal")
            signal_strength["bullish"] += 3
        elif rsi_data.get("signal") == "overbought":
            momentum_signals.append("bearish_reversal")
            signal_strength["bearish"] += 3

        # 多周期RSI动量
        multi_rsi = indicators.get("multi_rsi", {})
        if multi_rsi.get("overall_signal") == "strong_bullish":
            momentum_signals.append("strong_bullish_momentum")
            signal_strength["bullish"] += 4
        elif multi_rsi.get("overall_signal") == "strong_bearish":
            momentum_signals.append("strong_bearish_momentum")
            signal_strength["bearish"] += 4

        # MACD动量
        macd_data = indicators.get("macd", {})
        if macd_data.get("signal_type") == "bullish":
            momentum_signals.append("bullish_momentum")
            signal_strength["bullish"] += 2
        elif macd_data.get("signal_type") == "bearish":
            momentum_signals.append("bearish_momentum")
            signal_strength["bearish"] += 2

        # KDJ动量
        kdj_data = indicators.get("kdj", {})
        if kdj_data.get("signal") == "oversold":
            momentum_signals.append("bullish_reversal")
            signal_strength["bullish"] += 2
        elif kdj_data.get("signal") == "overbought":
            momentum_signals.append("bearish_reversal")
            signal_strength["bearish"] += 2

        # 威廉指标动量
        williams_data = indicators.get("williams_r", {})
        if williams_data.get("signal") == "oversold":
            momentum_signals.append("bullish_reversal")
            signal_strength["bullish"] += 2
        elif williams_data.get("signal") == "overbought":
            momentum_signals.append("bearish_reversal")
            signal_strength["bearish"] += 2

        # CCI动量
        cci_data = indicators.get("cci", {})
        if cci_data.get("signal") == "oversold":
            momentum_signals.append("bullish_reversal")
            signal_strength["bullish"] += 2
        elif cci_data.get("signal") == "overbought":
            momentum_signals.append("bearish_reversal")
            signal_strength["bearish"] += 2

        # MFI动量
        mfi_data = indicators.get("mfi", {})
        if mfi_data.get("signal") == "oversold":
            momentum_signals.append("bullish_reversal")
            signal_strength["bullish"] += 2
        elif mfi_data.get("signal") == "overbought":
            momentum_signals.append("bearish_reversal")
            signal_strength["bearish"] += 2

        # 成交量指标动量
        volume_data = indicators.get("volume_indicators", {})
        if volume_data.get("volume_signal") == "high_volume":
            momentum_signals.append("high_volume_momentum")
        elif volume_data.get("volume_signal") == "low_volume":
            momentum_signals.append("low_volume_momentum")

        # OBV动量
        obv_data = indicators.get("obv", {})
        if obv_data.get("trend") == "bullish":
            momentum_signals.append("bullish_volume_momentum")
            signal_strength["bullish"] += 1
        elif obv_data.get("trend") == "bearish":
            momentum_signals.append("bearish_volume_momentum")
            signal_strength["bearish"] += 1

        # VPT动量
        vpt_data = indicators.get("vpt", {})
        if vpt_data.get("trend") == "bullish":
            momentum_signals.append("bullish_price_volume_momentum")
            signal_strength["bullish"] += 1
        elif vpt_data.get("trend") == "bearish":
            momentum_signals.append("bearish_price_volume_momentum")
            signal_strength["bearish"] += 1

        # 高级指标动量
        advanced_data = indicators.get("advanced_indicators", {})

        # Ichimoku Cloud信号
        ichimoku = advanced_data.get("ichimoku", {})
        ichimoku_position = ichimoku.get("price_position", "unknown")
        if ichimoku_position == "above_cloud":
            momentum_signals.append("bullish_ichimoku_cloud")
            signal_strength["bullish"] += 3
        elif ichimoku_position == "below_cloud":
            momentum_signals.append("bearish_ichimoku_cloud")
            signal_strength["bearish"] += 3

        # DMI信号
        dmi_data = advanced_data.get("dmi", {})
        dmi_signal = dmi_data.get("signal", "neutral")
        adx_strength = dmi_data.get("trend_strength", 0)
        if dmi_signal == "strong_bullish":
            momentum_signals.append("strong_bullish_dmi")
            signal_strength["bullish"] += 4
        elif dmi_signal == "strong_bearish":
            momentum_signals.append("strong_bearish_dmi")
            signal_strength["bearish"] += 4
        elif dmi_signal == "bullish" and adx_strength > 20:
            momentum_signals.append("bullish_dmi")
            signal_strength["bullish"] += 2
        elif dmi_signal == "bearish" and adx_strength > 20:
            momentum_signals.append("bearish_dmi")
            signal_strength["bearish"] += 2

        # Aroon信号
        aroon_data = advanced_data.get("aroon", {})
        aroon_signal = aroon_data.get("signal", "neutral")
        if aroon_signal == "strong_bullish":
            momentum_signals.append("strong_bullish_aroon")
            signal_strength["bullish"] += 3
        elif aroon_signal == "strong_bearish":
            momentum_signals.append("strong_bearish_aroon")
            signal_strength["bearish"] += 3

        # Parabolic SAR信号
        sar_data = advanced_data.get("parabolic_sar", {})
        sar_signal = sar_data.get("signal", "neutral")
        if sar_signal == "bullish":
            momentum_signals.append("bullish_sar")
            signal_strength["bullish"] += 2
        elif sar_signal == "bearish":
            momentum_signals.append("bearish_sar")
            signal_strength["bearish"] += 2

        # Elder Ray信号
        elder_data = advanced_data.get("elder_ray", {})
        elder_signal = elder_data.get("signal", "neutral")
        if elder_signal == "strong_bullish":
            momentum_signals.append("strong_bullish_elder_ray")
            signal_strength["bullish"] += 3
        elif elder_signal == "strong_bearish":
            momentum_signals.append("strong_bearish_elder_ray")
            signal_strength["bearish"] += 3

        # Force Index信号
        force_data = advanced_data.get("force_index", {})
        force_signal = force_data.get("signal", "neutral")
        if force_signal == "strong_bullish":
            momentum_signals.append("strong_bullish_force")
            signal_strength["bullish"] += 3
        elif force_signal == "strong_bearish":
            momentum_signals.append("strong_bearish_force")
            signal_strength["bearish"] += 3

        # Chaikin Money Flow信号
        cmf_data = advanced_data.get("chaikin_money_flow", {})
        cmf_signal = cmf_data.get("signal", "neutral")
        if cmf_signal == "strong_bullish":
            momentum_signals.append("strong_bullish_money_flow")
            signal_strength["bullish"] += 3
        elif cmf_signal == "strong_bearish":
            momentum_signals.append("strong_bearish_money_flow")
            signal_strength["bearish"] += 3

        # A/D Line信号
        ad_data = advanced_data.get("ad_line", {})
        ad_trend = ad_data.get("trend", "neutral")
        if ad_trend == "bullish":
            momentum_signals.append("bullish_accumulation")
            signal_strength["bullish"] += 2
        elif ad_trend == "bearish":
            momentum_signals.append("bearish_distribution")
            signal_strength["bearish"] += 2

        # 波动率指标
        volatility_data = indicators.get("volatility_indicators", {})
        atr_data = volatility_data.get("atr", {})
        implied_vol_data = volatility_data.get("implied_volatility", {})

        # 波动率信号
        volatility_signal = implied_vol_data.get("signal", "normal")
        if volatility_signal == "fear_extreme":
            momentum_signals.append("extreme_fear_volatility")
            signal_strength["bearish"] += 2
        elif volatility_signal == "complacency":
            momentum_signals.append("low_volatility_complacency")
            signal_strength["bullish"] += 1

        # 市场广度指标
        breadth_data = indicators.get("breadth_indicators", {})
        mcclellan_data = breadth_data.get("mcclellan_oscillator", {})
        mcclellan_signal = mcclellan_data.get("signal", "neutral")
        if mcclellan_signal == "bullish":
            momentum_signals.append("bullish_market_breadth")
            signal_strength["bullish"] += 2
        elif mcclellan_signal == "bearish":
            momentum_signals.append("bearish_market_breadth")
            signal_strength["bearish"] += 2

        # 量化指标
        quant_data = indicators.get("quantitative_indicators", {})
        hurst_data = quant_data.get("hurst_exponent", {})
        hurst_signal = hurst_data.get("signal", "random_walk")
        if hurst_signal == "trending":
            momentum_signals.append("trending_market")
            signal_strength["bullish"] += 1
        elif hurst_signal == "mean_reverting":
            momentum_signals.append("mean_reverting_market")
            signal_strength["bearish"] += 1

        fractal_data = quant_data.get("fractal_dimension", {})
        fractal_signal = fractal_data.get("signal", "normal_complexity")
        if fractal_signal == "high_complexity":
            momentum_signals.append("high_complexity_pattern")
        elif fractal_signal == "low_complexity":
            momentum_signals.append("low_complexity_pattern")

        efficiency_data = quant_data.get("efficiency_ratio", {})
        efficiency_signal = efficiency_data.get("signal", "normal_efficiency")
        if efficiency_signal == "high_efficiency":
            momentum_signals.append("high_efficiency_trend")
            signal_strength["bullish"] += 2
        elif efficiency_signal == "low_efficiency":
            momentum_signals.append("low_efficiency_choppy")
            signal_strength["bearish"] += 1

        # 综合动量判断
        total_strength = signal_strength["bullish"] + signal_strength["bearish"]
        if total_strength == 0:
            overall_momentum = "neutral"
            momentum_confidence = 0
        else:
            bullish_ratio = signal_strength["bullish"] / total_strength
            bearish_ratio = signal_strength["bearish"] / total_strength

            # 更精细的动量分级
            if bullish_ratio > 0.75:
                overall_momentum = "very_strong_bullish"
                momentum_confidence = bullish_ratio
            elif bullish_ratio > 0.65:
                overall_momentum = "strong_bullish"
                momentum_confidence = bullish_ratio
            elif bullish_ratio > 0.55:
                overall_momentum = "moderate_bullish"
                momentum_confidence = bullish_ratio
            elif bearish_ratio > 0.75:
                overall_momentum = "very_strong_bearish"
                momentum_confidence = bearish_ratio
            elif bearish_ratio > 0.65:
                overall_momentum = "strong_bearish"
                momentum_confidence = bearish_ratio
            elif bearish_ratio > 0.55:
                overall_momentum = "moderate_bearish"
                momentum_confidence = bearish_ratio
            else:
                overall_momentum = "neutral"
                momentum_confidence = 0.5

        return {
            "overall_momentum": overall_momentum,
            "momentum_signals": momentum_signals,
            "signal_strength": signal_strength,
            "bullish_count": len([s for s in momentum_signals if "bullish" in s]),
            "bearish_count": len([s for s in momentum_signals if "bearish" in s]),
            "momentum_confidence": momentum_confidence,
            "advanced_signals_count": len([s for s in momentum_signals if any(word in s for word in ["ichimoku", "dmi", "aroon", "sar", "elder", "force", "cmf", "ad", "hurst", "fractal", "efficiency"])])
        }

    def _analyze_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析波动性"""
        if len(df) < 10:
            return {"volatility": 0, "volatility_level": "unknown"}

        closes = df['close'].values
        returns = np.diff(closes) / closes[:-1]

        # 计算波动率
        volatility = np.std(returns) * np.sqrt(252)  # 年化波动率

        # 波动率等级
        if volatility < 0.15:
            volatility_level = "low"
        elif volatility < 0.30:
            volatility_level = "medium"
        elif volatility < 0.50:
            volatility_level = "high"
        else:
            volatility_level = "very_high"

        return {
            "volatility": volatility,
            "volatility_level": volatility_level,
            "recent_volatility": np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else volatility
        }

    def _generate_trading_signal(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成交易信号

        Args:
            analysis_result: 技术分析结果

        Returns:
            交易信号
        """
        signals = []
        signal_details = {}

        # 趋势信号
        trend_data = analysis_result.get("trends", {})
        overall_trend = trend_data.get("overall_trend", "neutral")
        trend_strength = trend_data.get("strength", 0)

        if overall_trend == "bullish" and trend_strength > 40:
            signals.append(("bullish", trend_strength * 0.8))
            signal_details["trend_signal"] = f"上升趋势，强度{trend_strength:.1f}"
        elif overall_trend == "bearish" and trend_strength > 40:
            signals.append(("bearish", trend_strength * 0.8))
            signal_details["trend_signal"] = f"下降趋势，强度{trend_strength:.1f}"

        # 动量信号
        momentum_data = analysis_result.get("momentum", {})
        overall_momentum = momentum_data.get("overall_momentum", "neutral")
        momentum_confidence = momentum_data.get("momentum_confidence", 0)

        if overall_momentum == "strong_bullish" and momentum_confidence > 0.7:
            signals.append(("bullish", momentum_confidence * 100))
            signal_details["momentum_signal"] = f"强劲多头动量，置信度{momentum_confidence:.2f}"
        elif overall_momentum == "bullish" and momentum_confidence > 0.6:
            signals.append(("bullish", momentum_confidence * 80))
            signal_details["momentum_signal"] = f"多头动量，置信度{momentum_confidence:.2f}"
        elif overall_momentum == "strong_bearish" and momentum_confidence > 0.7:
            signals.append(("bearish", momentum_confidence * 100))
            signal_details["momentum_signal"] = f"强劲空头动量，置信度{momentum_confidence:.2f}"
        elif overall_momentum == "bearish" and momentum_confidence > 0.6:
            signals.append(("bearish", momentum_confidence * 80))
            signal_details["momentum_signal"] = f"空头动量，置信度{momentum_confidence:.2f}"

        # 技术指标信号
        indicators = analysis_result.get("indicators", {})

        # 移动平均线信号
        ma_data = indicators.get("moving_averages", {})
        ma_arrangement = ma_data.get("arrangement", "mixed")
        ema_data = indicators.get("ema", {})
        ema_arrangement = ema_data.get("arrangement", "mixed")

        if ma_arrangement == "bullish_alignment":
            signals.append(("bullish", 45))
            signal_details["ma_signal"] = "移动平均线多头排列"
        elif ma_arrangement == "bearish_alignment":
            signals.append(("bearish", 45))
            signal_details["ma_signal"] = "移动平均线空头排列"

        if ema_arrangement == "bullish_alignment":
            signals.append(("bullish", 50))
            signal_details["ema_signal"] = "指数移动平均线多头排列"
        elif ema_arrangement == "bearish_alignment":
            signals.append(("bearish", 50))
            signal_details["ema_signal"] = "指数移动平均线空头排列"

        # RSI信号
        rsi_data = indicators.get("rsi", {})
        rsi_value = rsi_data.get("rsi", 50)
        rsi_signal = rsi_data.get("signal", "neutral")

        if rsi_signal == "oversold":
            signals.append(("bullish", 75))
            signal_details["rsi_signal"] = f"RSI超卖（{rsi_value:.1f}），反弹机会"
        elif rsi_signal == "overbought":
            signals.append(("bearish", 75))
            signal_details["rsi_signal"] = f"RSI超买（{rsi_value:.1f}），回调风险"

        # 多周期RSI信号
        multi_rsi = indicators.get("multi_rsi", {})
        multi_rsi_signal = multi_rsi.get("overall_signal", "mixed")

        if multi_rsi_signal == "strong_bullish":
            signals.append(("bullish", 80))
            signal_details["multi_rsi_signal"] = "多周期RSI强劲看涨"
        elif multi_rsi_signal == "strong_bearish":
            signals.append(("bearish", 80))
            signal_details["multi_rsi_signal"] = "多周期RSI强劲看跌"

        # MACD信号
        macd_data = indicators.get("macd", {})
        macd_signal = macd_data.get("signal_type", "neutral")
        histogram = macd_data.get("histogram", 0)

        if macd_signal == "bullish" and histogram > 0:
            signals.append(("bullish", 60))
            signal_details["macd_signal"] = f"MACD多头排列（柱状图{histogram:.3f}）"
        elif macd_signal == "bearish" and histogram < 0:
            signals.append(("bearish", 60))
            signal_details["macd_signal"] = f"MACD空头排列（柱状图{histogram:.3f}）"

        # 布林带信号
        bb_data = indicators.get("bollinger_bands", {})
        bb_position = bb_data.get("position", "within_bands")

        if bb_position == "below_lower":
            signals.append(("bullish", 70))
            signal_details["bb_signal"] = "价格触及布林带下轨，超卖反弹机会"
        elif bb_position == "above_upper":
            signals.append(("bearish", 70))
            signal_details["bb_signal"] = "价格触及布林带上轨，超买回调风险"

        # 威廉指标信号
        williams_data = indicators.get("williams_r", {})
        williams_signal = williams_data.get("signal", "neutral")

        if williams_signal == "oversold":
            signals.append(("bullish", 65))
            signal_details["williams_signal"] = "威廉指标超卖"
        elif williams_signal == "overbought":
            signals.append(("bearish", 65))
            signal_details["williams_signal"] = "威廉指标超买"

        # 成交量确认信号
        volume_data = indicators.get("volume_indicators", {})
        volume_signal = volume_data.get("volume_signal", "normal")
        volume_ratio = volume_data.get("volume_ratio", 1.0)

        if volume_signal == "high_volume" and volume_ratio > 2.0:
            signal_details["volume_confirmation"] = f"高成交量确认（{volume_ratio:.1f}倍平均量）"
        elif volume_signal == "low_volume" and volume_ratio < 0.5:
            signal_details["volume_confirmation"] = f"低成交量（{volume_ratio:.1f}倍平均量）"

        # 综合信号分析
        if not signals:
            direction = "neutral"
            confidence = 50
            reasoning = "技术指标信号不明确，建议观望"
            risk_level = "medium"
        else:
            # 计算加权平均
            bullish_weight = sum(weight for signal, weight in signals if signal == "bullish")
            bearish_weight = sum(weight for signal, weight in signals if signal == "bearish")
            total_weight = bullish_weight + bearish_weight

            if total_weight == 0:
                direction = "neutral"
                confidence = 50
                reasoning = "无明确技术信号"
                risk_level = "medium"
            else:
                bullish_ratio = bullish_weight / total_weight
                bearish_ratio = bearish_weight / total_weight

                # 根据比例确定信号强度
                if bullish_ratio > 0.65:
                    direction = "bullish"
                    confidence = min(bullish_ratio * 100, 95)
                    reasoning = "技术指标整体看涨，建议逢低买入"
                    risk_level = "low" if confidence > 80 else "medium"
                elif bearish_ratio > 0.65:
                    direction = "bearish"
                    confidence = min(bearish_ratio * 100, 95)
                    reasoning = "技术指标整体看跌，建议逢高卖出"
                    risk_level = "low" if confidence > 80 else "medium"
                else:
                    direction = "neutral"
                    confidence = 50
                    reasoning = "多空力量均衡，建议观望等待明确信号"
                    risk_level = "medium"

        return {
            "direction": direction,
            "confidence": confidence,
            "reasoning": reasoning,
            "risk_level": risk_level,
            "signal_count": len(signals),
            "bullish_weight": bullish_weight if 'bullish_weight' in locals() else 0,
            "bearish_weight": bearish_weight if 'bearish_weight' in locals() else 0,
            "signal_details": signal_details,
            "detailed_signals": signals
        }

    def _generate_reasoning_report(self, analysis_result: Dict[str, Any], trading_signal: Dict[str, Any]) -> str:
        """生成推理报告"""
        report = []

        # 趋势分析
        trend_data = analysis_result.get("trends", {})
        report.append(f"趋势分析: {trend_data.get('overall_trend', '未知')}")
        report.append(f"趋势强度: {trend_data.get('strength', 0):.1f}")

        # 动量分析
        momentum_data = analysis_result.get("momentum", {})
        report.append(f"动量分析: {momentum_data.get('overall_momentum', '未知')}")

        # 技术指标汇总
        indicators = analysis_result.get("indicators", {})
        rsi = indicators.get("rsi", {}).get("rsi", 50)
        macd_type = indicators.get("macd", {}).get("signal_type", "neutral")

        report.append(f"RSI: {rsi:.1f}")
        report.append(f"MACD: {macd_type}")

        # 波动性
        volatility = analysis_result.get("volatility", {})
        report.append(f"波动率: {volatility.get('volatility_level', '未知')}")

        # 最终信号
        report.append(f"最终信号: {trading_signal.get('direction', 'neutral')}")
        report.append(f"信号置信度: {trading_signal.get('confidence', 50):.1f}%")

        return "\n".join(report)

    def get_required_fields(self) -> List[str]:
        """获取任务必需字段"""
        return ["ticker", "prices"]

    def _calculate_advanced_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算高级技术指标"""
        advanced = {}

        # Ichimoku Cloud (一目均衡表)
        advanced["ichimoku"] = self._calculate_ichimoku_cloud(df)

        # DMI (动向指标)
        advanced["dmi"] = self._calculate_dmi(df)

        # Aroon Indicator
        advanced["aroon"] = self._calculate_aroon(df)

        # Parabolic SAR
        advanced["parabolic_sar"] = self._calculate_parabolic_sar(df)

        # Elder Ray Index
        advanced["elder_ray"] = self._calculate_elder_ray(df)

        # Force Index
        advanced["force_index"] = self._calculate_force_index(df)

        # Trix Indicator
        advanced["trix"] = self._calculate_trix(df)

        # Ultimate Oscillator
        advanced["ultimate_oscillator"] = self._calculate_ultimate_oscillator(df)

        # Chaikin Money Flow
        advanced["chaikin_money_flow"] = self._calculate_chaikin_money_flow(df)

        # A/D Line (Accumulation/Distribution)
        advanced["ad_line"] = self._calculate_ad_line(df)

        return advanced

    def _calculate_ichimoku_cloud(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算一目均衡表"""
        if len(df) < 26:
            return {"signal": "insufficient_data"}

        # Tenkan-sen (转换线): 9日最高最低价平均
        tenkan_sen = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2

        # Kijun-sen (基准线): 26日最高最低价平均
        kijun_sen = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2

        # Senkou Span A (先行带A): (Tenkan-sen + Kijun-sen) / 2，前移26日
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

        # Senkou Span B (先行带B): 52日最高最低价平均，前移26日
        senkou_span_b = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)

        # Chikou Span (滞后线): 收盘价后移26日
        chikou_span = df['close'].shift(-26)

        current_price = df['close'].iloc[-1]
        current_tenkan = tenkan_sen.iloc[-1]
        current_kijun = kijun_sen.iloc[-1]
        current_senkou_a = senkou_span_a.iloc[-1]
        current_senkou_b = senkou_span_b.iloc[-1]
        current_chikou = chikou_span.iloc[-1]

        # 价格相对于云的位置
        if current_price > max(current_senkou_a, current_senkou_b):
            price_position = "above_cloud"
        elif current_price < min(current_senkou_a, current_senkou_b):
            price_position = "below_cloud"
        else:
            price_position = "inside_cloud"

        # 信号分析
        signals = []
        if current_tenkan > current_kijun:
            signals.append("bullish_tenkansen_kijunsen")
        if current_tenkan < current_kijun:
            signals.append("bearish_tenkansen_kijunsen")

        if current_senkou_a > current_senkou_b:
            signals.append("bullish_cloud")
        else:
            signals.append("bearish_cloud")

        return {
            "tenkan_sen": current_tenkan,
            "kijun_sen": current_kijun,
            "senkou_span_a": current_senkou_a,
            "senkou_span_b": current_senkou_b,
            "chikou_span": current_chikou,
            "price_position": price_position,
            "signals": signals,
            "cloud_thickness": abs(current_senkou_a - current_senkou_b)
        }

    def _calculate_dmi(self, df: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
        """计算动向指标(DMI)"""
        if len(df) < period + 1:
            return {"signal": "insufficient_data"}

        # 计算+DM, -DM, TR
        high_change = df['high'].diff()
        low_change = df['low'].diff()

        plus_dm = np.where((high_change > low_change) & (high_change > 0), high_change, 0)
        minus_dm = np.where((low_change > high_change) & (low_change > 0), -low_change, 0)

        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )

        # 计算平滑值
        plus_dm_smooth = pd.Series(plus_dm).rolling(window=period).mean()
        minus_dm_smooth = pd.Series(minus_dm).rolling(window=period).mean()
        tr_smooth = pd.Series(tr).rolling(window=period).mean()

        # 计算+DI, -DI
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth

        # 计算DX和ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        current_plus_di = plus_di.iloc[-1]
        current_minus_di = minus_di.iloc[-1]
        current_adx = adx.iloc[-1]

        # 信号判断
        if current_plus_di > current_minus_di:
            if current_adx > 25:
                signal = "strong_bullish"
            else:
                signal = "weak_bullish"
        elif current_minus_di > current_plus_di:
            if current_adx > 25:
                signal = "strong_bearish"
            else:
                signal = "weak_bearish"
        else:
            signal = "neutral"

        return {
            "plus_di": current_plus_di,
            "minus_di": current_minus_di,
            "adx": current_adx,
            "signal": signal,
            "trend_strength": current_adx
        }

    def _calculate_aroon(self, df: pd.DataFrame, period: int = 25) -> Dict[str, Any]:
        """计算Aroon指标"""
        if len(df) < period:
            return {"signal": "insufficient_data"}

        # 计算Aroon Up和Down
        high_period = df['high'].rolling(window=period).apply(lambda x: period - x.argmax() - 1)
        low_period = df['low'].rolling(window=period).apply(lambda x: period - x.argmin() - 1)

        aroon_up = 100 * high_period / (period - 1)
        aroon_down = 100 * low_period / (period - 1)

        current_aroon_up = aroon_up.iloc[-1]
        current_aroon_down = aroon_down.iloc[-1]

        # Aroon Oscillator
        aroon_oscillator = aroon_up - aroon_down

        # 信号判断
        if current_aroon_up > 70 and current_aroon_down < 30:
            signal = "strong_bullish"
        elif current_aroon_down > 70 and current_aroon_up < 30:
            signal = "strong_bearish"
        elif current_aroon_up > current_aroon_down:
            signal = "weak_bullish"
        elif current_aroon_down > current_aroon_up:
            signal = "weak_bearish"
        else:
            signal = "neutral"

        return {
            "aroon_up": current_aroon_up,
            "aroon_down": current_aroon_down,
            "aroon_oscillator": aroon_oscillator.iloc[-1],
            "signal": signal
        }

    def _calculate_parabolic_sar(self, df: pd.DataFrame, acceleration: float = 0.02, max_acceleration: float = 0.2) -> Dict[str, Any]:
        """计算抛物线SAR"""
        if len(df) < 5:
            return {"signal": "insufficient_data"}

        # 简化的SAR计算
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values

        # 初始化
        ep = closes[0]  # Extreme Point
        sar = [lows[0]]
        position = "long"  # 初始假设多头
        af = acceleration  # Acceleration Factor

        for i in range(1, len(closes)):
            if position == "long":
                if lows[i] > ep:
                    ep = lows[i]
                    af = min(af + acceleration, max_acceleration)
                sar_value = sar[i-1] + af * (ep - sar[i-1])

                if highs[i] >= sar_value:
                    position = "short"
                    ep = highs[i]
                    af = acceleration
            else:
                if highs[i] > ep:
                    ep = highs[i]
                    af = min(af + acceleration, max_acceleration)
                sar_value = sar[i-1] + af * (ep - sar[i-1])

                if lows[i] <= sar_value:
                    position = "long"
                    ep = lows[i]
                    af = acceleration

            sar.append(sar_value)

        current_price = closes[-1]
        current_sar = sar[-1]

        # 信号判断
        if current_price > current_sar:
            signal = "bullish"
        elif current_price < current_sar:
            signal = "bearish"
        else:
            signal = "neutral"

        return {
            "sar": current_sar,
            "signal": signal,
            "position": position
        }

    def _calculate_elder_ray(self, df: pd.DataFrame, period: int = 13) -> Dict[str, Any]:
        """计算艾达射线指标"""
        if len(df) < period:
            return {"signal": "insufficient_data"}

        # 计算EMA
        ema = df['close'].ewm(span=period).mean()

        # Bull Power和Bear Power
        bull_power = df['high'] - ema
        bear_power = df['low'] - ema

        current_bull = bull_power.iloc[-1]
        current_bear = bear_power.iloc[-1]

        # 信号判断
        if current_bull > 0 and current_bear > 0:
            signal = "strong_bullish"
        elif current_bull > 0 and current_bear < 0:
            signal = "bullish"
        elif current_bull < 0 and current_bear < 0:
            signal = "strong_bearish"
        elif current_bull < 0 and current_bear > 0:
            signal = "bearish"
        else:
            signal = "neutral"

        return {
            "bull_power": current_bull,
            "bear_power": current_bear,
            "signal": signal
        }

    def _calculate_force_index(self, df: pd.DataFrame, period: int = 13) -> Dict[str, Any]:
        """计算强力指数"""
        if len(df) < period + 1:
            return {"signal": "insufficient_data"}

        # 计算价格变化
        price_change = df['close'].diff()

        # Force Index = 价格变化 * 成交量
        force_index = price_change * df['volume']

        # 平滑
        force_index_ema = force_index.ewm(span=period).mean()

        current_force = force_index_ema.iloc[-1]

        # 信号判断
        if current_force > 0:
            if current_force > force_index_ema.quantile(0.75):
                signal = "strong_bullish"
            else:
                signal = "bullish"
        elif current_force < 0:
            if current_force < force_index_ema.quantile(0.25):
                signal = "strong_bearish"
            else:
                signal = "bearish"
        else:
            signal = "neutral"

        return {
            "force_index": current_force,
            "signal": signal
        }

    def _calculate_trix(self, df: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
        """计算Trix指标"""
        if len(df) < period * 3:
            return {"signal": "insufficient_data"}

        # 计算对数收益率
        log_returns = np.log(df['close'] / df['close'].shift(1))

        # 三重EMA
        ema1 = log_returns.ewm(span=period).mean()
        ema2 = ema1.ewm(span=period).mean()
        ema3 = ema2.ewm(span=period).mean()

        # Trix = EMA3的一阶差分
        trix = ema3.diff() * 10000  # 放大10000倍便于观察

        current_trix = trix.iloc[-1]
        trix_signal = trix.ewm(span=period).mean().iloc[-1]

        # 信号判断
        if current_trix > trix_signal:
            signal = "bullish"
        elif current_trix < trix_signal:
            signal = "bearish"
        else:
            signal = "neutral"

        return {
            "trix": current_trix,
            "trix_signal": trix_signal,
            "signal": signal
        }

    def _calculate_ultimate_oscillator(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算终极波动指标"""
        if len(df) < 28:
            return {"signal": "insufficient_data"}

        # BP = Close - Minimum(Low or Previous Close)
        bp = df['close'] - df[['low', 'close']].shift(1).min(axis=1)
        # TR = Maximum(High - Low, High - Previous Close, Previous Close - Low)
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)

        # 计算不同周期的Average
        avg7 = bp.rolling(window=7).sum() / tr.rolling(window=7).sum()
        avg14 = bp.rolling(window=14).sum() / tr.rolling(window=14).sum()
        avg28 = bp.rolling(window=28).sum() / tr.rolling(window=28).sum()

        # UO = 4*Avg7 + 2*Avg14 + Avg7
        uo = (4 * avg7 + 2 * avg14 + avg28) / 7 * 100

        current_uo = uo.iloc[-1]

        # 信号判断
        if current_uo > 70:
            signal = "overbought"
        elif current_uo < 30:
            signal = "oversold"
        elif current_uo > 50:
            signal = "bullish"
        elif current_uo < 50:
            signal = "bearish"
        else:
            signal = "neutral"

        return {
            "ultimate_oscillator": current_uo,
            "signal": signal
        }

    def _calculate_chaikin_money_flow(self, df: pd.DataFrame, period: int = 20) -> Dict[str, Any]:
        """计算蔡金资金流量"""
        if len(df) < period:
            return {"signal": "insufficient_data"}

        # Money Flow Multiplier
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfm = mfm.fillna(0)

        # Money Flow Volume
        mfv = mfm * df['volume']

        # Chaikin Money Flow
        cmf = mfv.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()

        current_cmf = cmf.iloc[-1]

        # 信号判断
        if current_cmf > 0.1:
            signal = "strong_bullish"
        elif current_cmf > 0:
            signal = "bullish"
        elif current_cmf < -0.1:
            signal = "strong_bearish"
        elif current_cmf < 0:
            signal = "bearish"
        else:
            signal = "neutral"

        return {
            "chaikin_money_flow": current_cmf,
            "signal": signal
        }

    def _calculate_ad_line(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算累积/派发线"""
        if len(df) < 2:
            return {"ad_line": 0, "trend": "neutral"}

        # CLV = (Close - Low) - (High - Close) / (High - Low)
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        clv = clv.fillna(0)

        # A/D Line
        ad_line = (clv * df['volume']).cumsum()

        current_ad = ad_line.iloc[-1]
        ad_ma = ad_line.rolling(window=20).mean().iloc[-1]

        # 趋势判断
        if current_ad > ad_ma * 1.05:
            trend = "bullish"
        elif current_ad < ad_ma * 0.95:
            trend = "bearish"
        else:
            trend = "neutral"

        return {
            "ad_line": current_ad,
            "ad_ma": ad_ma,
            "trend": trend
        }

    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算波动率指标"""
        volatility = {}

        # ATR (Average True Range)
        volatility["atr"] = self._calculate_atr(df)

        # VIX-like Volatility
        volatility["implied_volatility"] = self._calculate_implied_volatility(df)

        # Historical Volatility
        volatility["historical_volatility"] = self._calculate_historical_volatility(df)

        # Parkinson Volatility
        volatility["parkinson_volatility"] = self._calculate_parkinson_volatility(df)

        # Garman-Klass Volatility
        volatility["garman_klass_volatility"] = self._calculate_garman_klass_volatility(df)

        return volatility

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
        """计算平均真实波幅"""
        if len(df) < period:
            return {"atr": 0, "atr_ratio": 0}

        # True Range
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)

        # ATR
        atr = tr.rolling(window=period).mean()
        current_atr = atr.iloc[-1]

        # ATR Ratio
        atr_ratio = current_atr / df['close'].iloc[-1] if df['close'].iloc[-1] > 0 else 0

        # 波动率等级
        if atr_ratio > 0.05:
            volatility_level = "very_high"
        elif atr_ratio > 0.03:
            volatility_level = "high"
        elif atr_ratio > 0.015:
            volatility_level = "medium"
        else:
            volatility_level = "low"

        return {
            "atr": current_atr,
            "atr_ratio": atr_ratio,
            "volatility_level": volatility_level
        }

    def _calculate_implied_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算隐含波动率（基于历史数据模拟）"""
        if len(df) < 20:
            return {"implied_volatility": 0, "signal": "neutral"}

        # 简化的隐含波动率计算（实际应用中应使用期权定价模型）
        returns = df['close'].pct_change().dropna()
        historical_vol = returns.std() * np.sqrt(252)

        # 基于历史波动率模拟隐含波动率
        # 通常隐含波动率高于历史波动率
        volatility_premium = 0.1  # 10%的波动率溢价
        implied_vol = historical_vol * (1 + volatility_premium)

        # VIX-like signal
        if implied_vol > 0.4:
            signal = "fear_extreme"
        elif implied_vol > 0.3:
            signal = "fear"
        elif implied_vol < 0.15:
            signal = "complacency"
        else:
            signal = "normal"

        return {
            "implied_volatility": implied_vol,
            "signal": signal
        }

    def _calculate_historical_volatility(self, df: pd.DataFrame, period: int = 20) -> Dict[str, Any]:
        """计算历史波动率"""
        if len(df) < period:
            return {"historical_volatility": 0}

        returns = df['close'].pct_change().dropna()
        hist_vol = returns.rolling(window=period).std() * np.sqrt(252)

        return {
            "historical_volatility": hist_vol.iloc[-1],
            "volatility_trend": "increasing" if hist_vol.iloc[-1] > hist_vol.iloc[-5] else "decreasing"
        }

    def _calculate_parkinson_volatility(self, df: pd.DataFrame, period: int = 20) -> Dict[str, Any]:
        """计算Parkinson波动率"""
        if len(df) < period:
            return {"parkinson_volatility": 0}

        # Parkinson公式使用最高价和最低价
        high_low_ratio = np.log(df['high'] / df['low'])
        parkinson_vol = np.sqrt(high_low_ratio.rolling(window=period).mean() / (4 * np.log(2))) * np.sqrt(252)

        return {
            "parkinson_volatility": parkinson_vol.iloc[-1]
        }

    def _calculate_garman_klass_volatility(self, df: pd.DataFrame, period: int = 20) -> Dict[str, Any]:
        """计算Garman-Klass波动率"""
        if len(df) < period:
            return {"garman_klass_volatility": 0}

        # Garman-Klass公式
        log_hl = np.log(df['high'] / df['low'])
        log_co = np.log(df['close'] / df['open'])

        gk_vol = np.sqrt(
            (0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2).rolling(window=period).mean()
        ) * np.sqrt(252)

        return {
            "garman_klass_volatility": gk_vol.iloc[-1]
        }

    def _calculate_breadth_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算市场广度指标"""
        breadth = {}

        # McClellan Oscillator
        breadth["mcclellan_oscillator"] = self._calculate_mcclellan_oscillator(df)

        # Arms Index (TRIN)
        breadth["arms_index"] = self._calculate_arms_index(df)

        # New High-New Low Index
        breadth["new_high_low"] = self._calculate_new_high_low(df)

        return breadth

    def _calculate_mcclellan_oscillator(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算McClellan振荡器"""
        if len(df) < 39:
            return {"mcclellan_oscillator": 0, "signal": "neutral"}

        # 简化的McClellan计算（实际需要市场广度数据）
        # 这里使用价格变化作为代理
        advances = (df['close'] > df['close'].shift(1)).astype(int)
        declines = (df['close'] < df['close'].shift(1)).astype(int)

        # 19日EMA和39日EMA
        ema19_adv = advances.ewm(span=19).mean()
        ema39_adv = advances.ewm(span=39).mean()
        ema19_dec = declines.ewm(span=19).mean()
        ema39_dec = declines.ewm(span=39).mean()

        # McClellan Oscillator
        mco = (ema19_adv - ema39_adv) - (ema19_dec - ema39_dec)
        current_mco = mco.iloc[-1]

        # 信号判断
        if current_mco > 50:
            signal = "overbought"
        elif current_mco < -50:
            signal = "oversold"
        elif current_mco > 0:
            signal = "bullish"
        elif current_mco < 0:
            signal = "bearish"
        else:
            signal = "neutral"

        return {
            "mcclellan_oscillator": current_mco,
            "signal": signal
        }

    def _calculate_arms_index(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算ARMS指数(TRIN)"""
        if len(df) < 10:
            return {"arms_index": 1.0, "signal": "neutral"}

        # 简化的ARMS计算
        # ARMS = (Advancing Issues / Declining Issues) / (Advancing Volume / Declining Volume)
        advancing_volume = df[df['close'] > df['close'].shift(1)]['volume']
        declining_volume = df[df['close'] < df['close'].shift(1)]['volume']

        if len(advancing_volume) == 0 or len(declining_volume) == 0:
            return {"arms_index": 1.0, "signal": "neutral"}

        # 10日移动平均
        adv_vol_ma = advancing_volume.rolling(window=10).mean().iloc[-1] if len(advancing_volume) > 0 else 0
        dec_vol_ma = declining_volume.rolling(window=10).mean().iloc[-1] if len(declining_volume) > 0 else 0

        if dec_vol_ma == 0:
            arms_index = 1.0
        else:
            arms_index = adv_vol_ma / dec_vol_ma

        # 信号判断
        if arms_index < 0.8:
            signal = "bullish"
        elif arms_index > 1.2:
            signal = "bearish"
        else:
            signal = "neutral"

        return {
            "arms_index": arms_index,
            "signal": signal
        }

    def _calculate_new_high_low(self, df: pd.DataFrame, period: int = 52) -> Dict[str, Any]:
        """计算新高新低指数"""
        if len(df) < period:
            return {"new_high_low": 0, "signal": "neutral"}

        # 计算52周新高和新低
        high_52week = df['high'].rolling(window=period).max()
        low_52week = df['low'].rolling(window=period).min()

        new_highs = (df['high'] == high_52week).astype(int)
        new_lows = (df['low'] == low_52week).astype(int)

        # 新高新低指数
        nhl_index = (new_highs - new_lows).rolling(window=10).sum()
        current_nhl = nhl_index.iloc[-1]

        # 信号判断
        if current_nhl > 5:
            signal = "bullish"
        elif current_nhl < -5:
            signal = "bearish"
        else:
            signal = "neutral"

        return {
            "new_high_low": current_nhl,
            "signal": signal
        }

    def _calculate_quantitative_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算量化分析指标"""
        quantitative = {}

        # Hurst Exponent
        quantitative["hurst_exponent"] = self._calculate_hurst_exponent(df)

        # Fractal Dimension
        quantitative["fractal_dimension"] = self._calculate_fractal_dimension(df)

        # Shannon Entropy
        quantitative["shannon_entropy"] = self._calculate_shannon_entropy(df)

        # Efficiency Ratio
        quantitative["efficiency_ratio"] = self._calculate_efficiency_ratio(df)

        return quantitative

    def _calculate_hurst_exponent(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算Hurst指数"""
        if len(df) < 100:
            return {"hurst_exponent": 0.5, "signal": "neutral"}

        prices = df['close'].values

        # 简化的Hurst指数计算
        # 使用R/S分析方法
        n_values = [10, 20, 30, 40, 50]
        rs_values = []

        for n in n_values:
            if len(prices) < n:
                continue

            # 计算累积偏差
            mean_price = np.mean(prices[-n:])
            cumulative_deviation = np.cumsum(prices[-n:] - mean_price)

            # 计算极差
            r = np.max(cumulative_deviation) - np.min(cumulative_deviation)

            # 计算标准差
            s = np.std(prices[-n:])

            if s > 0:
                rs_values.append(r / s)

        if len(rs_values) == 0:
            return {"hurst_exponent": 0.5, "signal": "neutral"}

        # 回归计算Hurst指数
        log_n = np.log(n_values[:len(rs_values)])
        log_rs = np.log(rs_values)

        hurst = np.polyfit(log_n, log_rs, 1)[0]

        # 信号判断
        if hurst > 0.6:
            signal = "trending"  # 趋势性
        elif hurst < 0.4:
            signal = "mean_reverting"  # 均值回归
        else:
            signal = "random_walk"  # 随机游走

        return {
            "hurst_exponent": hurst,
            "signal": signal
        }

    def _calculate_fractal_dimension(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算分形维度"""
        if len(df) < 50:
            return {"fractal_dimension": 1.5, "signal": "neutral"}

        prices = df['close'].values

        # 使用盒维数方法计算分形维度
        scale_factors = [2, 4, 8, 16, 32]
        box_counts = []

        for scale in scale_factors:
            if len(prices) < scale:
                continue

            # 将价格序列分成盒子
            boxes = []
            box_size = len(prices) // scale

            for i in range(0, len(prices), box_size):
                box_prices = prices[i:i+box_size]
                if len(box_prices) > 0:
                    box_min = np.min(box_prices)
                    box_max = np.max(box_prices)
                    boxes.append((box_min, box_max))

            # 计算非空盒子数量
            unique_boxes = set()
            for box_min, box_max in boxes:
                unique_boxes.add(f"{box_min:.2f}-{box_max:.2f}")

            box_counts.append(len(unique_boxes))

        if len(box_counts) == 0:
            return {"fractal_dimension": 1.5, "signal": "neutral"}

        # 回归计算分形维度
        log_scale = np.log(scale_factors[:len(box_counts)])
        log_box = np.log(box_counts)

        fractal_dim = -np.polyfit(log_scale, log_box, 1)[0]

        # 信号判断
        if fractal_dim > 1.6:
            signal = "high_complexity"
        elif fractal_dim < 1.4:
            signal = "low_complexity"
        else:
            signal = "normal_complexity"

        return {
            "fractal_dimension": fractal_dim,
            "signal": signal
        }

    def _calculate_shannon_entropy(self, df: pd.DataFrame, bins: int = 10) -> Dict[str, Any]:
        """计算香农熵"""
        if len(df) < 20:
            return {"shannon_entropy": 0, "signal": "neutral"}

        prices = df['close'].values

        # 计算价格变化
        returns = np.diff(prices) / prices[:-1]

        # 分箱
        hist, bin_edges = np.histogram(returns, bins=bins)
        probabilities = hist / len(returns)

        # 计算香农熵
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))

        # 信号判断
        if entropy > 2.0:
            signal = "high_uncertainty"
        elif entropy < 1.5:
            signal = "low_uncertainty"
        else:
            signal = "normal_uncertainty"

        return {
            "shannon_entropy": entropy,
            "signal": signal
        }

    def _calculate_efficiency_ratio(self, df: pd.DataFrame, period: int = 10) -> Dict[str, Any]:
        """计算效率比率"""
        if len(df) < period:
            return {"efficiency_ratio": 0.5, "signal": "neutral"}

        prices = df['close'].values[-period:]

        # 净价格变化
        net_change = abs(prices[-1] - prices[0])

        # 总价格变化
        total_change = sum(abs(prices[i] - prices[i-1]) for i in range(1, len(prices)))

        if total_change == 0:
            efficiency_ratio = 0.5
        else:
            efficiency_ratio = net_change / total_change

        # 信号判断
        if efficiency_ratio > 0.7:
            signal = "high_efficiency"
        elif efficiency_ratio < 0.3:
            signal = "low_efficiency"
        else:
            signal = "normal_efficiency"

        return {
            "efficiency_ratio": efficiency_ratio,
            "signal": signal
        }