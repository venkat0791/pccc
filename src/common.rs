//! Types needed in multiple modules

/// Enumeration of binary symbol values
#[derive(Clone, Eq, PartialEq, Debug, Copy)]
pub enum Bit {
    /// Binary symbol `0`
    Zero = 0,
    /// Binary symbol `1`
    One = 1,
}

/// Custom error type
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// Invalid input error
    #[error("{0}")]
    InvalidInput(String),
    /// File read/write error
    #[error("{0}")]
    FileReadWriteError(#[from] std::io::Error),
    /// Serde read/write error
    #[error("{0}")]
    SerdeReadWriteError(#[from] serde_json::Error),
    /// Unknown error
    #[error("Unknown error")]
    Unknown,
}
