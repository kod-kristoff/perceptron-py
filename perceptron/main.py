import numpy as np
import pandas as pd


def main():
    owls_and_albatrosses()
    condors_and_albatrosses()


def owls_and_albatrosses():
    print(">>> owls_and_albatrosses")
    X, y = generate_owls_and_albatrosses()
    np.savetxt("albaowl_data.csv", X, delimiter=",", fmt="%.18e")
    np.savetxt("albaowl_species.csv", y, delimiter=",", fmt="%d")
    w, errors = fit(X, y, eta=0.01, n_iter=200)
    np.savetxt("albaowl_perceptron_weights.csv", w, delimiter=",", fmt="%.18e")
    y_pred = predict(X, w)
    num_correct_predictions = (y_pred == y).sum()
    accuracy = (num_correct_predictions / y.shape[0]) * 100
    print('Perceptron accuracy: %.2f%%' % accuracy)


def condors_and_albatrosses():
    print(">>> condors_and_albatrosses")
    X, y = generate_condors_and_albatrosses()
    np.savetxt("albacondor_data.csv", X, delimiter=",", fmt="%.18e")
    np.savetxt("albacondor_species.csv", y, delimiter=",", fmt="%d")
    w, errors = fit(X, y, eta=0.01, n_iter=200)
    np.savetxt("albacondor_perceptron_weights.csv", w, delimiter=",", fmt="%.18e")
    y_pred = predict(X, w)
    num_correct_predictions = (y_pred == y).sum()
    accuracy = (num_correct_predictions / y.shape[0]) * 100
    print('Perceptron accuracy: %.2f%%' % accuracy)


def generate_owls_and_albatrosses():
    aX, ay = species_generator(9000.0, 800.0, 300.0, 20.0, 100, 1, 100)
    albatross_dic = {
        "weight-(g)": aX[:, 0],
        "wingspan-(cm)": aX[:, 1],
        "species": ay,
    }
    albatross_df = pd.DataFrame(albatross_dic)

    oX, oy = species_generator(1000.0, 200.0, 100.0, 15.0, 100, -1, 100)
    owl_dic = {
        "weight-(g)": oX[:, 0],
        "wingspan-(cm)": oX[:, 1],
        "species": oy,
    }
    owl_df = pd.DataFrame(owl_dic)

    df = albatross_df.append(owl_df, ignore_index=True)
    # df.to_csv("owls_and_albatrosses.csv", index=False)

    df_shuffle = df.sample(frac=1, random_state=1).reset_index(drop=True)
    X = df_shuffle[['weight-(g)','wingspan-(cm)']].to_numpy()
    y = df_shuffle['species'].to_numpy()
    return X, y


def generate_condors_and_albatrosses():
    aX, ay = species_generator(9000.0, 800.0, 300.0, 20.0, 100, 1, 100)
    albatross_dic = {
        "weight-(g)": aX[:, 0],
        "wingspan-(cm)": aX[:, 1],
        "species": ay,
    }
    albatross_df = pd.DataFrame(albatross_dic)

    cX, cy = species_generator(12000.0, 1000.0, 290.0, 15.0, 100, -1, 100)
    condor_dic = {
        "weight-(g)": cX[:, 0],
        "wingspan-(cm)": cX[:, 1],
        "species": cy,
    }
    condor_df = pd.DataFrame(condor_dic)

    df = albatross_df.append(condor_df, ignore_index=True)
    df.to_csv("condors_and_albatrosses.csv", index=False)

    df_shuffle = df.sample(frac=1, random_state=1).reset_index(drop=True)
    X = df_shuffle[['weight-(g)','wingspan-(cm)']].to_numpy()
    y = df_shuffle['species'].to_numpy()
    return X, y



def fit(X, y, eta=0.001, n_iter=100):
    errors = []
    w = random_weights(X, random_state=1)
    for exemplar in range(n_iter):
        error = 0
        for xi, target in zip(X, y):
            delta = eta * (target - predict(xi, w))
            w[1:] += delta * xi
            w[0] += delta
            error += int(delta != 0.0)
        errors.append(error)
    return w, errors


def predict(X, w):
    return np.where(net_input(X, w) < 0.0, -1, 1)


def net_input(X, w):
    return np.dot(X, w[1:]) + w[0]


def random_weights(X, random_state: int):
    rand = np.random.RandomState(random_state)
    return rand.normal(loc=0.0, scale=0.01, size=1+X.shape[1])


def species_generator(mu1, sigma1, mu2, sigma2, n_samples,target, seed: int):
    rand = np.random.RandomState(seed)
    f1 = rand.normal(mu1, sigma1, n_samples)
    f2 = rand.normal(mu2, sigma2, n_samples)
    X = np.array([f1, f2])
    X = X.transpose()
    y = np.full((n_samples,), target)
    return X, y


if __name__ == "__main__":
    main()
