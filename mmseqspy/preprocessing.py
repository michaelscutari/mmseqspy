from .logger import logger


def clean(df, sequence_col="sequence", valid_amino_acids="ACDEFGHIKLMNPQRSTVWY"):
    """
    Removes sequences with invalid protein characters.

    Parameters:
        df (pd.DataFrame): Input DataFrame with protein sequences.
        sequence_col (str): Name of the column containing sequences.
        valid_amino_acids (str): String of valid amino acid characters.

    Returns:
        pd.DataFrame: Cleaned DataFrame with only valid sequences.
    """
    logger.info(
        f"Cleaning sequences in column '{sequence_col}' with valid amino acids: {valid_amino_acids}"
    )
    logger.info(f"Input dataframe has {len(df)} sequences")

    # Make a copy to avoid modifying the original
    df = df.copy()

    # Early return for empty DataFrames
    if len(df) == 0:
        return df

    # Handle missing values first
    df = df.dropna(subset=[sequence_col])
    logger.debug(f"After removing NaN values: {len(df)} sequences")

    # Convert to uppercase
    df[sequence_col] = df[sequence_col].str.upper()

    # Filter out empty sequences
    df = df[df[sequence_col].str.len() > 0]

    # Filter sequences with invalid amino acids
    valid_sequence_mask = df[sequence_col].apply(
        lambda seq: all(aa in valid_amino_acids for aa in seq)
    )

    result_df = df[valid_sequence_mask].reset_index(drop=True)

    invalid_count = len(df) - len(result_df)
    logger.info(f"Removed {invalid_count} sequences with invalid amino acids")
    logger.info(f"Final dataframe has {len(result_df)} valid sequences")

    return result_df
