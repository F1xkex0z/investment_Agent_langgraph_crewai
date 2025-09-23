"""
调试导入问题的脚本
"""

import sys
import os

print("Python路径:")
for i, path in enumerate(sys.path):
    print(f"{i}: {path}")

print("\n" + "="*50)

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, current_dir)
sys.path.insert(0, src_dir)

print(f"当前目录: {current_dir}")
print(f"源码目录: {src_dir}")

print("\n更新后的Python路径:")
for i, path in enumerate(sys.path[:5]):  # 只显示前5个
    print(f"{i}: {path}")

print("\n" + "="*50)

# 测试导入
print("测试模块导入:")

try:
    import config
    print(f"✅ config导入成功")
    print(f"   VERSION: {getattr(config, 'VERSION', 'N/A')}")
    print(f"   PROJECT_NAME: {getattr(config, 'PROJECT_NAME', 'N/A')}")
except Exception as e:
    print(f"❌ config导入失败: {e}")

try:
    from utils.logging_config import setup_logger
    print("✅ logging_config导入成功")
    logger = setup_logger("debug_test")
    logger.info("测试日志")
except Exception as e:
    print(f"❌ logging_config导入失败: {e}")

try:
    from utils.shared_context import get_global_context
    print("✅ shared_context导入成功")
    context = get_global_context()
    context.set("debug", "test", source_agent="debug")
    print(f"   上下文测试: {context.get('debug')}")
except Exception as e:
    print(f"❌ shared_context导入失败: {e}")

try:
    from utils.data_processing import get_data_processor
    print("✅ data_processing导入成功")
    processor = get_data_processor()
    test_data = processor.clean_numeric_data("1,234.56")
    print(f"   数据处理测试: {test_data}")
except Exception as e:
    print(f"❌ data_processing导入失败: {e}")

try:
    from agents.base_agent import BaseAgent
    print("✅ base_agent导入成功")
except Exception as e:
    print(f"❌ base_agent导入失败: {e}")

try:
    from tasks.base_task import BaseTask
    print("✅ base_task导入成功")
except Exception as e:
    print(f"❌ base_task导入失败: {e}")

print("\n" + "="*50)
print("导入测试完成")