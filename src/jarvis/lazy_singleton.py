"""
Thread-safe lazy singleton helper for service instantiation.

This module provides a reusable pattern for creating singleton services
that are initialized only when first accessed, with thread safety guarantees.
"""
from __future__ import annotations
import threading
from typing import Callable, TypeVar, Generic, Any

T = TypeVar("T")


class LazySingleton(Generic[T]):
    """
    A thread-safe lazy singleton wrapper.
    
    This class provides:
    - Lazy initialization: The wrapped instance is created only on first access
    - Thread safety: Uses double-checked locking to ensure only one instance
    - Transparent proxy: All attribute access is forwarded to the wrapped instance
    - Type preservation: Works with type checkers and IDE autocomplete
    
    Example:
        class MyService:
            def __init__(self):
                print("Expensive initialization...")
                self.data = load_data()
            
            def process(self, item):
                return self.data.transform(item)
        
        # Create a lazy singleton instance
        my_service = LazySingleton(MyService)
        
        # The service is not initialized yet
        # First access triggers initialization
        result = my_service.process("test")  # Prints "Expensive initialization..."
        
        # Subsequent access uses the same instance
        result2 = my_service.process("test2")  # No initialization
    """
    
    def __init__(self, factory: Callable[[], T]):
        """
        Initialize the lazy singleton wrapper.
        
        Args:
            factory: A callable that returns an instance of type T.
                    Usually this is just the class itself.
        """
        self._factory = factory
        self._lock = threading.Lock()
        self._instance: T | None = None
    
    def __getattr__(self, name: str) -> Any:
        """
        Proxy attribute access to the wrapped instance.
        
        This method is called when accessing any attribute on the LazySingleton.
        It ensures the instance is created (if not already) and forwards the
        attribute access to the real instance.
        
        Args:
            name: The attribute name being accessed
            
        Returns:
            The attribute from the wrapped instance
        """
        if self._instance is None:  # First check (fast path)
            with self._lock:
                if self._instance is None:  # Second check (under lock)
                    self._instance = self._factory()
        return getattr(self._instance, name)
    
    def __call__(self, *args, **kwargs):
        """
        Make the singleton callable if the wrapped class is callable.
        
        This allows patterns like:
            service = LazySingleton(MyCallableService)
            result = service(arg1, arg2)  # If MyCallableService defines __call__
        """
        # Ensure instance exists
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = self._factory()
        return self._instance(*args, **kwargs)
    
    @property
    def instance(self) -> T:
        """
        Get the actual wrapped instance explicitly.
        
        This property ensures the instance is created and returns it directly.
        Useful for debugging or when you need the actual object reference.
        
        Returns:
            The wrapped singleton instance
        """
        # Force initialization by accessing any attribute
        _ = self.__getattr__("__class__")
        assert self._instance is not None  # Type narrowing for mypy
        return self._instance
    
    def is_initialized(self) -> bool:
        """
        Check if the singleton instance has been created yet.
        
        Returns:
            True if the instance has been created, False otherwise
        """
        return self._instance is not None
    
    def _reset_instance(self) -> None:
        """
        Reset the singleton instance (for testing purposes only).
        
        WARNING: This method is intended for testing only and should not
        be used in production code. It can lead to inconsistent state if
        there are existing references to the old instance.
        """
        with self._lock:
            self._instance = None