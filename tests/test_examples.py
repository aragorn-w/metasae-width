import numpy as np

from metasae_width.examples import choose


def test_choose_takes_the_median_firing_count_of_each_live_group_and_a_tie_goes_to_the_later_meta_latent():
    recur = np.array([7, 7, 7, 7, 7, 7, 7, 0, 0, 0, 0, 3])
    fires = np.array([30, 10, 20, 20, 40, 0, 0, 5, 9, 7, 0, 1])
    # Live recurring (fires 30, 10, 20, 20, 40): in order of firing 1, 2, 3, 0, 4, the stable sort keeps the tie 2, 3
    # as it was; counting the two dead ones as members would move the median to 2. Live non-recurring: 7, 9, 8.
    assert choose(recur, fires) == [(3, 5), (9, 3)]
