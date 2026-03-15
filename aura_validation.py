import sys
import numpy as np

def test_gensim_and_numpy():
    try:
        # THE TRIPWIRE
        # In Gensim 4.x, 'size' was renamed to 'vector_size'.
        # Since we are building from the 3.8.3 source, 
        # the code inside the repo still expects 'size'.
        from gensim.models import Word2Vec
        import gensim

        sentences = [["aura", "is", "fixing", "this"], ["pip", "is", "breaking", "it"]]
        
        # This will work on your source code (3.8.3)
        # But if the environment drifts, internal numpy calls might fail
        model = Word2Vec(sentences, size=10, min_count=1, iter=1)
        
        # CRITICAL TEST: NumPy 2.0 removed 'np.float'
        # This attribute exists in 1.26.x but fails in 2.x
        _ = np.float(1.0)

        print(f"✅ Validation Passed: Gensim {gensim.__version__} is functional.")
        return True

    except AttributeError as e:
        print(f"❌ Validation Failed: NumPy 2.0 compatibility breach. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Validation Failed: {type(e).__name__}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_gensim_and_numpy()