def handle_exception(logger):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"An unexpected error occurred: {e}", exc_info=True)
                raise
        return wrapper
    return decorator