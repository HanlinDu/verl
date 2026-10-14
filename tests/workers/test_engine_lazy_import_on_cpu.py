"""Runnable stdlib-only checks of real initializer/registry with synthetic siblings.

No optional engine, training, Ray or GPU execution is claimed by this check.
Run: python tests/workers/test_engine_lazy_import_on_cpu.py
"""

import ast
import builtins
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "verl.workers.engine"


@contextmanager
def sandbox():
    saved_env = {name: os.environ.pop(name, None) for name in ("VERL_ENGINE_DEVICE", "VERL_ENGINE_VENDOR")}
    saved = {name: module for name, module in sys.modules.items() if name == "verl" or name.startswith("verl.")}
    for name in saved:
        del sys.modules[name]
    for name in ("verl", "verl.workers", PREFIX):
        module = ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    engine = sys.modules[PREFIX]
    base = ModuleType(PREFIX + ".base")
    base.__package__ = PREFIX
    base.BaseEngine = type("BaseEngine", (), {})
    base.os = os
    base.get_device_name = lambda: "cuda"
    base.get_vendor = lambda: "nvidia"
    tree = ast.parse((ROOT / "verl/workers/engine/base.py").read_text())
    registry_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "EngineRegistry")
    exec(compile(ast.Module(body=[registry_node], type_ignores=[]), "actual_registry", "exec"), base.__dict__)
    sys.modules[base.__name__] = base
    registry = base.EngineRegistry
    fsdp = ModuleType(PREFIX + ".fsdp")
    fsdp.FSDPEngine = type("FSDPEngine", (base.BaseEngine,), {})
    fsdp.FSDPEngineWithLMHead = type("FSDPEngineWithLMHead", (fsdp.FSDPEngine,), {})
    registry.register("language_model", ["fsdp", "fsdp2"])(fsdp.FSDPEngineWithLMHead)
    sys.modules[fsdp.__name__] = fsdp
    original_import = builtins.__import__

    def guard(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 1 and name in {"torchtitan", "veomni", "automodel", "mindspeed", "megatron"}:
            raise AssertionError("unselected eager import: " + name)
        return original_import(name, globals, locals, fromlist, level)

    try:
        builtins.__import__ = guard
        exec(
            compile((ROOT / "verl/workers/engine/__init__.py").read_text(), "actual_initializer", "exec"),
            engine.__dict__,
        )
        builtins.__import__ = original_import
        attempts, loaded, failures = [], {}, {}

        def fake_import(name, package):
            assert package == PREFIX
            key = name.removeprefix(".")
            attempts.append(key)
            if key in failures:
                raise failures[key]
            if key not in loaded:
                module = ModuleType(PREFIX + name)
                for export in engine._OPTIONAL_EXPORTS[key]:
                    setattr(module, export, type(export, (base.BaseEngine,), {}))
                if key == "mindspeed":
                    registry.register("language_model", "megatron", device="npu")(module.MindspeedEngineWithLMHead)
                    registry.register("language_model", "mindspeed_megatron", device="npu")(
                        module.MindSpeedMegatronEngineWithLMHead
                    )
                else:
                    registry.register("language_model", key)(getattr(module, engine._OPTIONAL_EXPORTS[key][1]))
                loaded[key] = module
            return loaded[key]

        engine.import_module = fake_import
        yield engine, registry, attempts, loaded, failures
    finally:
        builtins.__import__ = original_import
        for name in list(sys.modules):
            if name == "verl" or name.startswith("verl."):
                del sys.modules[name]
        sys.modules.update(saved)
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def check():
    with sandbox() as (engine, registry, attempts, loaded, failures):
        assert registry.get_engine_cls("language_model", "fsdp2") is engine.FSDPEngineWithLMHead
        assert attempts == []
        assert engine.__all__ == ["BaseEngine", "EngineRegistry", "FSDPEngine", "FSDPEngineWithLMHead"]
        cls = registry.get_engine_cls("language_model", "automodel")
        assert cls is registry.get_engine_cls("language_model", "automodel")
        assert isinstance(registry.new("language_model", "automodel"), cls)
        assert engine.AutomodelEngineWithLMHead is cls and attempts == ["automodel"]
        assert engine.VeOmniEngine is loaded["veomni"].VeOmniEngine
        assert "veomni" in loaded
        custom = type("Custom", (engine.BaseEngine,), {})
        registry.register("language_model", "custom")(custom)
        before = list(attempts)
        assert registry.get_engine_cls("language_model", "custom") is custom
        for model, backend, message in [
            ("missing", "megatron", "Unknown model_type"),
            ("language_model", "missing", "Unknown backend"),
        ]:
            try:
                registry.get_engine_cls(model, backend)
            except AssertionError as error:
                assert message in str(error)
            else:
                raise AssertionError("unknown lookup passed")
        assert attempts == before
    for error in (ImportError("missing dependency"), IndexError("actual runtime failure")):
        with sandbox() as (engine, registry, attempts, loaded, failures):
            failures["automodel"] = error
            try:
                registry.get_engine_cls("language_model", "automodel")
            except type(error) as caught:
                assert caught is error
            else:
                raise AssertionError("selected backend failure hidden")
            if isinstance(error, ImportError):
                assert engine.AutomodelEngine is None
            else:
                try:
                    _ = engine.AutomodelEngine
                except type(error) as caught:
                    assert caught is error
                else:
                    raise AssertionError("explicit export runtime failure hidden")
    with sandbox() as (engine, registry, attempts, loaded, failures):
        cls = registry.get_engine_cls("language_model", "megatron")
        assert cls is engine.MegatronEngineWithLMHead
        assert attempts == ["mindspeed", "megatron"]
        old = os.environ.get("VERL_ENGINE_DEVICE")
        try:
            os.environ["VERL_ENGINE_DEVICE"] = "npu"
            assert (
                registry.get_engine_cls("language_model", "megatron") is loaded["mindspeed"].MindspeedEngineWithLMHead
            )
            assert (
                registry.get_engine_cls("language_model", "mindspeed_megatron")
                is loaded["mindspeed"].MindSpeedMegatronEngineWithLMHead
            )
        finally:
            if old is None:
                os.environ.pop("VERL_ENGINE_DEVICE", None)
            else:
                os.environ["VERL_ENGINE_DEVICE"] = old
        assert attempts == ["mindspeed", "megatron"]
    with sandbox() as (engine, registry, attempts, loaded, failures):
        failures["mindspeed"] = ImportError("optional mindspeed unavailable")
        assert registry.get_engine_cls("language_model", "megatron") is engine.MegatronEngineWithLMHead
        assert attempts == ["mindspeed", "megatron"]
    with sandbox() as (engine, registry, attempts, loaded, failures):
        os.environ["VERL_ENGINE_DEVICE"] = "npu"
        assert (
            registry.get_engine_cls("language_model", "mindspeed_megatron") is engine.MindSpeedMegatronEngineWithLMHead
        )
        assert attempts == ["mindspeed"]
    with sandbox() as (engine, registry, attempts, loaded, failures):
        error = ImportError("selected megatron failed after another device registered")
        failures["megatron"] = error
        for _ in range(2):
            try:
                registry.get_engine_cls("language_model", "megatron")
            except ImportError as caught:
                assert caught is error
            else:
                raise AssertionError("partial registration hid selected backend failure")
    with sandbox() as (engine, registry, attempts, loaded, failures):
        registry.get_engine_cls("language_model", "automodel")
        os.environ["VERL_ENGINE_DEVICE"] = "cpu"
        try:
            registry.get_engine_cls("language_model", "automodel")
        except ValueError as error:
            assert "No engine registered" in str(error)
        else:
            raise AssertionError("unsupported device lookup passed")
        assert attempts == ["automodel"]
    for error in (ImportError("failed after registration"), IndexError("late runtime failure")):
        with sandbox() as (engine, registry, attempts, loaded, failures):
            original = engine.import_module

            def late_failure(name, package, wrapped=original, failure=error):
                module = wrapped(name, package)
                if name == ".automodel":
                    raise failure
                return module

            engine.import_module = late_failure
            for _ in range(2):
                try:
                    registry.get_engine_cls("language_model", "automodel")
                except type(error) as caught:
                    assert caught is error
                else:
                    raise AssertionError("registered class from failed import accepted")
            assert attempts == ["automodel"]
    print("engine lazy-import CPU checks PASS (synthetic siblings, actual initializer/registry)")


if __name__ == "__main__":
    check()
