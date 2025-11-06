"""Clear the plot rendering cache by calling the internal cache clear method."""
import sys
sys.path.insert(0, 'C:\\Coding\\SAI')

from app.main import _render_plot_bytes

# Clear the LRU cache
_render_plot_bytes.cache_clear()
print("Plot cache cleared successfully!")
print(f"Cache info after clear: {_render_plot_bytes.cache_info()}")
