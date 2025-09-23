"""
数据处理和格式转换工具
提供数据清洗、转换和标准化功能
"""

import json
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import re

from .logging_config import get_logger


class DataProcessor:
    """数据处理器，提供各种数据转换和清洗功能"""

    def __init__(self):
        self.logger = get_logger("data_processor")

    def pandas_to_dict_records(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        将Pandas DataFrame转换为字典记录列表

        Args:
            df: Pandas DataFrame

        Returns:
            字典记录列表
        """
        if df is None or df.empty:
            return []

        try:
            # 处理时间类型数据
            result = []
            for _, row in df.iterrows():
                record = {}
                for col, value in row.items():
                    # 处理NaN值
                    if pd.isna(value):
                        record[col] = None
                    # 处理时间类型
                    elif pd.api.types.is_datetime64_any_dtype(df[col]):
                        record[col] = value.isoformat() if hasattr(value, 'isoformat') else str(value)
                    # 处理数值类型
                    elif pd.api.types.is_numeric_dtype(df[col]):
                        record[col] = float(value) if not pd.isna(value) else None
                    else:
                        record[col] = str(value) if value is not None else None
                result.append(record)
            return result
        except Exception as e:
            self.logger.error(f"DataFrame转字典失败: {e}")
            return []

    def dict_to_pandas(self, data: Union[List[Dict[str, Any]], Dict[str, Any]]) -> pd.DataFrame:
        """
        将字典数据转换为Pandas DataFrame

        Args:
            data: 字典数据

        Returns:
            Pandas DataFrame
        """
        if not data:
            return pd.DataFrame()

        try:
            if isinstance(data, dict):
                return pd.DataFrame([data])
            elif isinstance(data, list):
                return pd.DataFrame(data)
            else:
                self.logger.warning(f"不支持的数据类型: {type(data)}")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"字典转DataFrame失败: {e}")
            return pd.DataFrame()

    def clean_numeric_data(self, data: Any) -> Any:
        """
        清理数值数据，移除特殊字符并转换为适当类型

        Args:
            data: 原始数据

        Returns:
            清理后的数据
        """
        if isinstance(data, (int, float)):
            return data

        if isinstance(data, str):
            # 移除常见的数值格式字符
            cleaned = re.sub(r'[,%\s+]', '', data)
            try:
                if '.' in cleaned:
                    return float(cleaned)
                else:
                    return int(cleaned)
            except ValueError:
                return data

        if isinstance(data, dict):
            return {k: self.clean_numeric_data(v) for k, v in data.items()}

        if isinstance(data, list):
            return [self.clean_numeric_data(item) for item in data]

        return data

    def format_financial_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化财务指标数据

        Args:
            metrics: 原始财务指标

        Returns:
            格式化的财务指标
        """
        if not metrics:
            return {}

        formatted = {}
        for key, value in metrics.items():
            # 特殊处理某些指标
            if key in ['revenue', 'net_income', 'assets', 'liabilities']:
                formatted[key] = {
                    'value': self.clean_numeric_data(value),
                    'unit': 'CNY',
                    'formatted': self.format_currency(value)
                }
            elif key in ['pe_ratio', 'pb_ratio', 'roe', 'roa', 'debt_ratio']:
                formatted[key] = {
                    'value': self.clean_numeric_data(value),
                    'unit': 'ratio',
                    'formatted': self.format_percentage(value, as_ratio=True)
                }
            elif key in ['market_cap']:
                formatted[key] = {
                    'value': self.clean_numeric_data(value),
                    'unit': 'CNY',
                    'formatted': self.format_large_number(value)
                }
            else:
                formatted[key] = value

        return formatted

    def format_currency(self, value: Any, currency: str = 'CNY') -> str:
        """
        格式化货币数值

        Args:
            value: 数值
            currency: 货币类型

        Returns:
            格式化字符串
        """
        try:
            num_value = float(self.clean_numeric_data(value))
            if abs(num_value) >= 1e9:
                return f"{currency} {num_value/1e9:.2f}B"
            elif abs(num_value) >= 1e6:
                return f"{currency} {num_value/1e6:.2f}M"
            elif abs(num_value) >= 1e3:
                return f"{currency} {num_value/1e3:.2f}K"
            else:
                return f"{currency} {num_value:,.2f}"
        except (ValueError, TypeError):
            return str(value)

    def format_percentage(self, value: Any, as_ratio: bool = False) -> str:
        """
        格式化百分比

        Args:
            value: 数值
            as_ratio: 是否作为比率显示

        Returns:
            格式化字符串
        """
        try:
            num_value = float(self.clean_numeric_data(value))
            if as_ratio:
                return f"{num_value:.2f}"
            else:
                return f"{num_value * 100:.2f}%"
        except (ValueError, TypeError):
            return str(value)

    def format_large_number(self, value: Any) -> str:
        """
        格式化大数字

        Args:
            value: 数值

        Returns:
            格式化字符串
        """
        try:
            num_value = float(self.clean_numeric_data(value))
            if abs(num_value) >= 1e12:
                return f"{num_value/1e12:.2f}T"
            elif abs(num_value) >= 1e9:
                return f"{num_value/1e9:.2f}B"
            elif abs(num_value) >= 1e6:
                return f"{num_value/1e6:.2f}M"
            elif abs(num_value) >= 1e3:
                return f"{num_value/1e3:.2f}K"
            else:
                return f"{num_value:,.2f}"
        except (ValueError, TypeError):
            return str(value)

    def calculate_technical_indicators(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        计算技术指标

        Args:
            price_data: 价格数据DataFrame

        Returns:
            技术指标字典
        """
        if price_data is None or price_data.empty or 'close' not in price_data.columns:
            return {}

        try:
            indicators = {}
            closes = price_data['close'].values

            if len(closes) < 2:
                return indicators

            # 移动平均线
            indicators['ma_5'] = self._calculate_ma(closes, 5)
            indicators['ma_10'] = self._calculate_ma(closes, 10)
            indicators['ma_20'] = self._calculate_ma(closes, 20)

            # RSI
            indicators['rsi'] = self._calculate_rsi(closes)

            # MACD
            macd_data = self._calculate_macd(closes)
            indicators.update(macd_data)

            # 布林带
            bb_data = self._calculate_bollinger_bands(closes)
            indicators.update(bb_data)

            # 波动率
            indicators['volatility'] = self._calculate_volatility(closes)

            return indicators

        except Exception as e:
            self.logger.error(f"计算技术指标失败: {e}")
            return {}

    def _calculate_ma(self, data: np.ndarray, period: int) -> float:
        """计算移动平均线"""
        if len(data) < period:
            return np.nan
        return np.mean(data[-period:])

    def _calculate_rsi(self, data: np.ndarray, period: int = 14) -> float:
        """计算RSI"""
        if len(data) < period + 1:
            return np.nan

        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """计算MACD"""
        if len(data) < slow:
            return {'macd': np.nan, 'signal': np.nan, 'histogram': np.nan}

        ema_fast = self._calculate_ema(data, fast)
        ema_slow = self._calculate_ema(data, slow)
        macd_line = ema_fast - ema_slow

        return {
            'macd': macd_line,
            'signal': macd_line,  # 简化版本，实际需要计算signal line
            'histogram': macd_line
        }

    def _calculate_ema(self, data: np.ndarray, period: int) -> float:
        """计算指数移动平均线"""
        if len(data) < period:
            return np.nan

        alpha = 2 / (period + 1)
        ema = data[0]
        for price in data[1:]:
            ema = alpha * price + (1 - alpha) * ema

        return ema

    def _calculate_bollinger_bands(self, data: np.ndarray, period: int = 20, std_dev: float = 2) -> Dict[str, float]:
        """计算布林带"""
        if len(data) < period:
            return {'upper_band': np.nan, 'middle_band': np.nan, 'lower_band': np.nan}

        middle_band = np.mean(data[-period:])
        std = np.std(data[-period:])
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)

        return {
            'upper_band': upper_band,
            'middle_band': middle_band,
            'lower_band': lower_band
        }

    def _calculate_volatility(self, data: np.ndarray, period: int = 20) -> float:
        """计算波动率"""
        if len(data) < period:
            return np.nan

        returns = np.diff(data[-period:]) / data[-period:-1]
        return np.std(returns) * np.sqrt(252)  # 年化波动率

    def merge_agent_outputs(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        合并多个智能体输出

        Args:
            outputs: 智能体输出列表

        Returns:
            合并后的输出
        """
        if not outputs:
            return {}

        merged = {
            'timestamp': datetime.now().isoformat(),
            'agent_count': len(outputs),
            'agents': []
        }

        # 收集所有智能体输出
        for output in outputs:
            if isinstance(output, dict):
                agent_name = output.get('agent_name', 'unknown')
                merged['agents'].append({
                    'name': agent_name,
                    'output': output
                })

                # 合并特定字段
                for key in ['signal', 'confidence', 'reasoning']:
                    if key in output and key not in merged:
                        merged[key] = output[key]

        return merged

    def format_agent_message(self, agent_name: str, content: Any, **kwargs) -> Dict[str, Any]:
        """
        格式化智能体消息

        Args:
            agent_name: 智能体名称
            content: 消息内容
            **kwargs: 其他参数

        Returns:
            格式化的消息
        """
        message = {
            'agent_name': agent_name,
            'timestamp': datetime.now().isoformat(),
            'content': content,
        }

        # 添加可选字段
        optional_fields = ['signal', 'confidence', 'reasoning', 'metadata']
        for field in optional_fields:
            if field in kwargs:
                message[field] = kwargs[field]

        return message

    def validate_stock_data(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        验证股票数据完整性

        Args:
            data: 股票数据

        Returns:
            (是否有效, 错误信息列表)
        """
        errors = []
        required_fields = ['ticker', 'prices', 'start_date', 'end_date']

        for field in required_fields:
            if field not in data or data[field] is None:
                errors.append(f"缺少必需字段: {field}")

        # 验证价格数据
        if 'prices' in data and isinstance(data['prices'], list):
            if len(data['prices']) == 0:
                errors.append("价格数据为空")
            else:
                # 检查必要的价格字段
                sample_price = data['prices'][0]
                price_fields = ['close', 'open', 'high', 'low', 'volume']
                for field in price_fields:
                    if field not in sample_price:
                        errors.append(f"价格数据缺少字段: {field}")

        return len(errors) == 0, errors

    def safe_json_loads(self, json_str: str) -> Any:
        """
        安全的JSON解析

        Args:
            json_str: JSON字符串

        Returns:
            解析结果或None
        """
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析失败: {e}")
            return None

    def safe_json_dumps(self, obj: Any, **kwargs) -> str:
        """
        安全的JSON序列化

        Args:
            obj: 要序列化的对象
            **kwargs: JSON序列化参数

        Returns:
            JSON字符串
        """
        try:
            return json.dumps(obj, ensure_ascii=False, **kwargs)
        except (TypeError, ValueError) as e:
            self.logger.error(f"JSON序列化失败: {e}")
            return "{}"

    def normalize_ticker(self, ticker: str) -> str:
        """
        标准化股票代码

        Args:
            ticker: 股票代码

        Returns:
            标准化的股票代码
        """
        if not ticker:
            return ""

        # 移除空格和特殊字符
        normalized = re.sub(r'[\s\.]', '', ticker.upper())

        # 确保是6位数字（A股代码格式）
        if normalized.isdigit() and len(normalized) <= 6:
            normalized = normalized.zfill(6)

        return normalized

    def get_date_range(self, days_back: int = 365) -> Tuple[str, str]:
        """
        获取日期范围

        Args:
            days_back: 向前的天数

        Returns:
            (开始日期, 结束日期)
        """
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=days_back)

        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


# 全局数据处理器实例
data_processor = DataProcessor()


def get_data_processor() -> DataProcessor:
    """获取全局数据处理器实例"""
    return data_processor