#!/usr/bin/env python
"""A2S2D Playground 测试脚本

验证 A2S2D playground 的核心功能：
1. 配置加载
2. 模块导入
3. Playground 注册
4. 阶段 Exp 创建
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """测试模块导入"""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)

    errors = []

    # Test A2S2D library imports
    try:
        from playground.a2s2d.lib import (
            ProjectConfig,
            VariableMetadata,
            VariableRole,
            CandidateRelation,
            ModelResult,
            AuditResult,
        )
        print("✓ A2S2D types imported")
    except Exception as e:
        errors.append(f"A2S2D types: {e}")

    try:
        from playground.a2s2d.lib.config import ProjectConfig
        from playground.a2s2d.lib.io import load_data, load_codebook
        from playground.a2s2d.lib.metadata import build_metadata
        print("✓ A2S2D core functions imported")
    except Exception as e:
        errors.append(f"A2S2D core: {e}")

    # Test Exp imports
    try:
        from playground.a2s2d.core.exp import (
            A2S2DBaseExp,
            InputExp,
            IdentificationExp,
            CandidateExp,
            ModelDesignExp,
            ModelExecExp,
            AuditExp,
            RobustnessExp,
            LiteratureExp,
        )
        print("✓ Stage Exps imported")
    except Exception as e:
        errors.append(f"Stage Exps: {e}")

    # Test Playground import
    try:
        from playground.a2s2d.core.playground import A2S2DPlayground
        print("✓ A2S2DPlayground imported")
    except Exception as e:
        errors.append(f"A2S2DPlayground: {e}")

    # Test registry
    try:
        from evomaster.core.registry import get_registry_info
        info = get_registry_info()
        if 'a2s2d' in info:
            print(f"✓ a2s2d registered: {info['a2s2d']}")
        else:
            errors.append("a2s2d not in registry")
    except Exception as e:
        errors.append(f"Registry: {e}")

    if errors:
        print("\n❌ Import errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\n✅ All imports successful")
        return True


def test_config_loading():
    """测试配置加载"""
    print("\n" + "=" * 60)
    print("Testing config loading...")
    print("=" * 60)

    errors = []

    try:
        import yaml
        from pathlib import Path

        config_path = project_root / "configs" / "a2s2d" / "config.yaml"
        if not config_path.exists():
            errors.append(f"Config file not found: {config_path}")
        else:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Check required fields
            required = ["llm", "agents", "session", "inputs", "modeling", "audit"]
            missing = [k for k in required if k not in config]

            if missing:
                errors.append(f"Missing config sections: {missing}")
            else:
                print(f"✓ Config loaded: {config_path}")
                print(f"  - LLM providers: {list(config.get('llm', {}).keys())}")
                print(f"  - Agents: {list(config.get('agents', {}).keys())}")
                print(f"  - Pipeline stages: {len(config.get('pipeline', {}).get('stages', []))}")

    except Exception as e:
        errors.append(str(e))

    if errors:
        print("\n❌ Config loading errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\n✅ Config loaded successfully")
        return True


def test_prompts():
    """测试提示词文件"""
    print("\n" + "=" * 60)
    print("Testing prompt files...")
    print("=" * 60)

    prompts_dir = project_root / "playground" / "a2s2d" / "prompts"

    required_prompts = [
        "variable_identification_system.txt",
        "variable_identification_user.txt",
        "model_selection_system.txt",
        "model_selection_user.txt",
        "literature_system.txt",
        "literature_user.txt",
    ]

    missing = []
    for prompt_file in required_prompts:
        prompt_path = prompts_dir / prompt_file
        if prompt_path.exists():
            size = prompt_path.stat().st_size
            print(f"✓ {prompt_file} ({size} bytes)")
        else:
            missing.append(prompt_file)

    if missing:
        print(f"\n❌ Missing prompt files: {missing}")
        return False
    else:
        print("\n✅ All prompt files present")
        return True


def test_exp_creation():
    """测试 Exp 创建"""
    print("\n" + "=" * 60)
    print("Testing Exp creation...")
    print("=" * 60)

    errors = []

    try:
        from playground.a2s2d.core.exp import InputExp, A2S2DBaseExp
        from pathlib import Path

        # Test creating a simple Exp
        exp = InputExp(
            agent=None,
            config=None,
            run_dir=Path("/tmp/test_a2s2d"),
            state={},
        )

        print(f"✓ InputExp created: {exp.exp_name}")
        print(f"  - run_dir: {exp.run_dir}")
        print(f"  - state: {exp.state}")

    except Exception as e:
        errors.append(str(e))

    if errors:
        print("\n❌ Exp creation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\n✅ Exp creation successful")
        return True


def main():
    """运行所有测试"""
    print("A2S2D Playground Test Suite")
    print("=" * 60)

    results = {
        "imports": test_imports(),
        "config": test_config_loading(),
        "prompts": test_prompts(),
        "exp_creation": test_exp_creation(),
    }

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed!")
        print("=" * 60)
        return 0
    else:
        print("❌ Some tests failed")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
