import sys
import numpy as np
import os

def test_gensim_source():
    try:
        # 1. Check if we are actually using the local source (3.8.3)
        import gensim
        print(f"Current Gensim version: {gensim.__version__}")
        
        from gensim.models import Word2Vec
        sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
        
        # This parameter 'size' exists in 3.8.3 but is 'vector_size' in 4.x
        # If this line fails with TypeError, pip installed 4.x instead of your source.
        model = Word2Vec(sentences, size=10, min_count=1, iter=1)
        
        # 2. THE TRIPWIRE: NumPy 2.0 attribute check
        # This exists in NumPy 1.26.4 but is GONE in NumPy 2.x
        _ = np.float(1.0)

        print("✅ Validation Passed: Local source is functional and NumPy is compatible.")
        return True

    except TypeError as e:
        print(f"❌ Validation Failed: API Mismatch. Is this Gensim 4.x? {e}")
        sys.exit(1)
    except AttributeError as e:
        print(f"❌ Validation Failed: NumPy 2.0+ Compatibility Breach. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Validation Failed: {type(e).__name__}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_gensim_source()