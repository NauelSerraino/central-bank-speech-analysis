import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
from scipy.stats import entropy

array = np.arange(0.01, 1, 0.01)
bern_entropy = {p: (bernoulli.entropy(p))**2 for p in array}
df = pd.DataFrame(bern_entropy.items(), columns=['p', 'entropy'])

# Entropy - Bernoulli distribution
plt.plot(df['p'], df['entropy'])
plt.xlabel('p')
plt.ylabel('Entropy')
plt.title('Entropy of Bernoulli distribution')
plt.show()
# breakpoint()

def bernoulli_entropy(p):
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

array = np.arange(0.01, 1, 0.01)
bern_entropy = {p: bernoulli_entropy(p) for p in array}
df = pd.DataFrame(bern_entropy.items(), columns=['p', 'entropy'])

# Entropy - Bernoulli distribution
plt.plot(df['p'], df['entropy'])
plt.xlabel('p')
plt.ylabel('Entropy')
plt.title('Entropy of Bernoulli distribution (base 2)')
plt.show()


