import sys
import os
import numpy as np

def test_gensim_source_runtime():
    try:
        # THE TRIPWIRE
        # If AURA or Pip pushes NumPy to 2.0+, this internal import or 
        # the np.float call below will trigger an AttributeError.
        from gensim.models import Word2Vec
        import gensim
        
        sentences = [["machine", "learning", "is", "cool"]]
        
        # This confirms we are running the local 3.x source code
        # In 4.x, 'size' is 'vector_size'
        model = Word2Vec(sentences, size=10, min_count=1, iter=1)
        
        # Legacy attribute check (Removed in NumPy 2.0)
        _ = np.float(1.0)

        print(f"✅ Validation Passed: Local Gensim source (v{gensim.__version__}) functional.")
        return True

    except AttributeError as e:
        print(f"❌ Validation Failed: NumPy 2.0+ Compatibility Breach. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Validation Failed: {type(e).__name__}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_gensim_source_runtime()