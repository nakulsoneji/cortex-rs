#[macro_export]
macro_rules! matrix {
	() => {
		$crate::linalg::Matrix::zeros(0, 0)
	};

	([$($first:expr),* $(,)?] $(, [$($rest:expr),* $(,)?])* $(,)?) => {{
		let rows = 1usize $(+ { let _ = &[$($rest),*]; 1usize })*;
		let cols = <[()]>::len(&[$({ let _ = &$first; () }),*]);

		$(
			let row_len = <[()]>::len(&[$({ let _ = &$rest; () }),*]);
			assert_eq!(
				row_len,
				cols,
				"All rows must have the same number of columns"
			);
		)*

		let data = vec![
			$($first as f32),*
			$(, $($rest as f32),*)*
		];

		$crate::linalg::Matrix::new(data, [rows, cols])
	}};
}
