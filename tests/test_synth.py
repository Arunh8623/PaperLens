from paperlens.synth import generate_papers, messy_stream, expected_final, topic_of, TOPICS


def test_same_seed_same_output_and_different_seed_differs():
    a, b, c = generate_papers(50, 1), generate_papers(50, 1), generate_papers(50, 2)
    assert a == b and a != c
    assert [p.to_json() for p in a] == [p.to_json() for p in b]


def test_all_valid_unique_and_synthetic():
    ps = generate_papers(300, 3)
    assert len({p.key for p in ps}) == 300
    assert all(p.synthetic for p in ps)
    assert all(p.arxiv_id.startswith("synth.") for p in ps)


def test_topics_are_recoverable_mostly():
    ps = generate_papers(200, 4)
    # topic words dominate abstracts, so the guess should be stable and diverse
    topics = {topic_of(p) for p in ps}
    assert topics == set(TOPICS)


def test_messy_stream_counts_and_ground_truth():
    base = generate_papers(200, 5)
    s = messy_stream(base, seed=5)
    assert len(s) == 200 + 20 + 20 + 10
    final = expected_final(s)
    assert len(final) == 200                            # one version per paper
    revised = [p for p in final.values() if p.title.endswith(" revised")]
    assert len(revised) == 20                           # newer versions won
    assert not any(p.title.endswith(" draft") for p in final.values())  # stale lost
    assert messy_stream(base, seed=5) == s              # reproducible
