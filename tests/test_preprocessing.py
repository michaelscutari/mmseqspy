from mmseqspy import clean


def test_clean_valid_sequences(fluorescence_data):
    """Test cleaning with valid protein sequences."""
    # Make a copy of data to avoid modifying the fixture
    df = fluorescence_data.copy()

    # Add some invalid sequences
    df.loc[0, "sequence"] = df.loc[0, "sequence"] + "X"  # Invalid amino acid
    df.loc[1, "sequence"] = df.loc[
        1, "sequence"
    ].lower()  # Should be converted to uppercase
    df.loc[2, "sequence"] = None  # Null value

    # Run cleaning
    cleaned_df = clean(df, sequence_col="sequence")

    # Check results
    assert (
        len(cleaned_df) == len(df) - 2
    )  # Should remove the invalid and null sequences
    assert (
        cleaned_df["sequence"].str.isupper().all()
    )  # All sequences should be uppercase
    assert cleaned_df["sequence"].notna().all()  # No null values

    # Check that only valid amino acids remain
    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    for seq in cleaned_df["sequence"]:
        assert all(aa in valid_amino_acids for aa in seq)
