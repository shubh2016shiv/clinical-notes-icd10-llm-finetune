"""
core/services — pure domain services with no infrastructure dependencies.

All classes here depend only on core/models and core/ports.
Concrete infrastructure (LLM clients, file loaders) must be injected.
"""
