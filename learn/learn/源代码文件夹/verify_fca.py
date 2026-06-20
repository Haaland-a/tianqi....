import sys
import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_BASE_DIR, '..', 'ultralytics-main'))

from ultralytics import YOLO
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def verify_fca_module():
    print("FCA-CL 模块集成验证")
    print("=" * 60)
    
    print("\n[1] 检查模块导入...")
    try:
        from ultralytics.nn.modules import FCA, FCACLModule
        print("✓ FCA 模块导入成功")
        print("✓ FCACLModule 导入成功")
    except ImportError as e:
        print(f"✗ 模块导入失败: {e}")
        return False
    
    print("\n[2] 检查模型配置...")
    model_yaml = os.path.join(_BASE_DIR, '..', 'ultralytics-main', 'ultralytics', 'cfg', 'models', 'v8', 'yolov8-fca.yaml')
    
    if not os.path.exists(model_yaml):
        print(f"✗ 配置文件不存在: {model_yaml}")
        return False
    print(f"✓ 配置文件存在: {model_yaml}")
    
    print("\n[3] 加载模型并检查结构...")
    try:
        model = YOLO(model_yaml, task='detect')
        print("✓ 模型加载成功")
        
        print("\n" + "=" * 60)
        print("模型结构信息:")
        print("=" * 60)
        model.info()
        
        print("\n" + "=" * 60)
        print("检查 FCA 层:")
        print("=" * 60)
        
        fca_found = False
        for name, module in model.model.named_modules():
            if 'FCA' in module.__class__.__name__:
                print(f"✓ 发现 FCA 层: {name}")
                print(f"  - 类型: {module.__class__.__name__}")
                print(f"  - 参数量: {sum(p.numel() for p in module.parameters()):,}")
                fca_found = True
        
        if not fca_found:
            print("✗ 未发现 FCA 层!")
            return False
        
        print("\n" + "=" * 60)
        print("参数量统计:")
        print("=" * 60)
        
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        
        print("\n与标准 YOLOv8n 对比:")
        std_model = YOLO('yolov8n.yaml')
        std_params = sum(p.numel() for p in std_model.model.parameters())
        
        param_increase = total_params - std_params
        param_increase_pct = (param_increase / std_params) * 100
        
        print(f"  标准 YOLOv8n: {std_params:,} 参数")
        print(f"  YOLOv8-FCA:   {total_params:,} 参数")
        print(f"  增加: {param_increase:,} 参数 ({param_increase_pct:.2f}%)")
        
        if param_increase_pct < 5:
            print("✓ 参数量增加符合预期 (<5%)")
        else:
            print("⚠ 参数量增加较多，请检查配置")
        
        print("\n" + "=" * 60)
        print("测试前向传播:")
        print("=" * 60)
        
        test_input = torch.randn(1, 3, 640, 640)
        try:
            with torch.no_grad():
                output = model.model(test_input)
            print("✓ 前向传播成功")
            print(f"  - 输入尺寸: {test_input.shape}")
            if isinstance(output, (list, tuple)):
                print(f"  - 输出数量: {len(output)}")
                for i, out in enumerate(output):
                    print(f"  - 输出{i+1}尺寸: {out.shape}")
        except Exception as e:
            print(f"✗ 前向传播失败: {e}")
            return False
        
        print("\n" + "=" * 60)
        print("✓✓✓ 验证完成！FCA-CL 模块已成功集成 ✓✓✓")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = verify_fca_module()
    
    if success:
        print("\n✅ 您可以开始训练模型了!")
        print("   运行命令: python yolo_train.py")
    else:
        print("\n❌ 验证失败，请检查:")
        print("   1. fca_cl.py 文件是否正确创建")
        print("   2. __init__.py 是否正确修改")
        print("   3. yolov8-fca.yaml 配置是否正确")
