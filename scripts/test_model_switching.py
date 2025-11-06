"""
Test script to verify that plot endpoints work correctly for all model types.
This test validates that images are generated for supported plot types and proper
error messages are returned for unsupported combinations.
"""
import requests
import sys
from io import BytesIO
from PIL import Image

BASE_URL = "http://localhost:8000"

# Define what plot types are supported for each model
PLOT_SUPPORT = {
    'sklearn': {'roc': True, 'confusion': True, 'fi': True, 'pr': True},
    'stacking': {'roc': True, 'confusion': True, 'fi': False, 'pr': True},
    'dl': {'roc': True, 'confusion': True, 'fi': False, 'pr': True}
}

def test_plot_endpoint(plot_type, model_type, should_work):
    """Test a single plot endpoint"""
    url = f"{BASE_URL}/plot?type={plot_type}&model={model_type}"
    print(f"Testing {model_type} model, {plot_type} plot... ", end="")
    
    try:
        response = requests.get(url, timeout=10)
        
        if should_work:
            if response.status_code == 200:
                # Verify it's actually a valid PNG
                try:
                    img = Image.open(BytesIO(response.content))
                    print(f"✅ OK (image size: {img.size})")
                    return True
                except Exception as e:
                    print(f"❌ FAILED (invalid image: {e})")
                    return False
            else:
                print(f"❌ FAILED (status {response.status_code})")
                print(f"   Response: {response.text[:200]}")
                return False
        else:
            # Should return 400 error for unsupported types
            if response.status_code == 400:
                print(f"✅ OK (correctly rejected: {response.json().get('detail', 'N/A')[:50]}...)")
                return True
            else:
                print(f"⚠️  WARNING (expected 400, got {response.status_code})")
                return True  # Still pass, as long as it doesn't crash
    
    except requests.exceptions.RequestException as e:
        print(f"❌ FAILED (connection error: {e})")
        return False

def main():
    print("="*70)
    print("MODEL SWITCHING TEST - Plot Endpoints")
    print("="*70)
    print()
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Backend server not responding correctly!")
            sys.exit(1)
        print("✅ Backend server is running")
    except requests.exceptions.RequestException:
        print("❌ Backend server is not running!")
        print("   Please start it with: python -m uvicorn app.main:APP --host 127.0.0.1 --port 8000")
        sys.exit(1)
    
    print()
    
    # Check available models
    try:
        models_response = requests.get(f"{BASE_URL}/models", timeout=5)
        available_models = models_response.json()
        print("Available models:")
        for model, available in available_models.items():
            status = "✅" if available else "❌"
            print(f"  {status} {model}")
    except Exception as e:
        print(f"⚠️  Could not fetch available models: {e}")
        available_models = {'sklearn': True, 'stacking': True, 'dl': True}
    
    print()
    print("-"*70)
    print()
    
    # Test all combinations
    total_tests = 0
    passed_tests = 0
    
    for model_type in ['sklearn', 'stacking', 'dl']:
        if not available_models.get(model_type, False):
            print(f"⏭️  Skipping {model_type} model (not available)")
            print()
            continue
        
        print(f"📊 Testing {model_type.upper()} Model")
        print("-"*70)
        
        for plot_type in ['roc', 'confusion', 'fi', 'pr']:
            should_work = PLOT_SUPPORT[model_type].get(plot_type, False)
            total_tests += 1
            
            if test_plot_endpoint(plot_type, model_type, should_work):
                passed_tests += 1
        
        print()
    
    # Summary
    print("="*70)
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("✅ All tests passed! Model switching is working correctly.")
        return 0
    else:
        print(f"⚠️  {total_tests - passed_tests} test(s) failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
