use rawml::prelude::*;

fn main() {
    test_decompositions();
    test_math();
    println!("\n✓ All tests passed!");
}

fn assert_matrix_approx(a: &Matrix, b: &Matrix, tol: f32, name: &str) {
    assert_eq!(a.shape, b.shape, "{}: shape mismatch", name);
    for r in 0..a.shape[0] {
        for c in 0..a.shape[1] {
            let av = unsafe { *a.get_unchecked(r, c) };
            let bv = unsafe { *b.get_unchecked(r, c) };
            assert!((av - bv).abs() < tol, 
                "{}: mismatch at ({},{}) got {} expected {}", name, r, c, av, bv);
        }
    }
}

fn assert_vector_approx(a: &Vector, b: &Vector, tol: f32, name: &str) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch", name);
    for i in 0..a.len() {
        let av = unsafe { *a.get_unchecked(i) };
        let bv = unsafe { *b.get_unchecked(i) };
        assert!((av - bv).abs() < tol,
            "{}: mismatch at {} got {} expected {}", name, i, av, bv);
    }
}

fn assert_approx(a: f32, b: f32, tol: f32, name: &str) {
    assert!((a - b).abs() < tol, "{}: got {} expected {}", name, a, b);
}

fn test_decompositions() {
    println!("Testing decompositions...");

    {
        let a = Matrix::new(vec![2.0, 1.0, 1.0, 3.0], [2, 2]);
        let lu = a.lu().expect("lu failed");

        let reconstructed: Matrix = lu.l.dot(&lu.u);
        assert_approx(unsafe { *lu.l.get_unchecked(0, 0) }, 1.0, 1e-5, "L diagonal 0");
        assert_approx(unsafe { *lu.l.get_unchecked(1, 1) }, 1.0, 1e-5, "L diagonal 1");
        assert_approx(unsafe { *lu.l.get_unchecked(0, 1) }, 0.0, 1e-5, "L upper zero");
        assert_approx(unsafe { *lu.u.get_unchecked(1, 0) }, 0.0, 1e-5, "U lower zero");
        println!("  ✓ LU");
    }

    {
        let a = Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]);
        let qr = a.qr().expect("qr failed");

        let qtq = qr.q.transpose().dot(&qr.q);
        let identity = Matrix::eye(2);
        assert_matrix_approx(&qtq, &identity, 1e-4, "QR: QtQ = I");

        let reconstructed = qr.q.dot(&qr.r);
        assert_matrix_approx(&reconstructed, &a, 1e-4, "QR: Q*R = A");
        println!("  ✓ QR");
    }

    {
        let a = Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]);
        let svd = a.svd().expect("svd failed");

        assert!(svd.s.data[0] > svd.s.data[1], "SVD: s not decreasing");
        assert!(svd.s.data[1] > 0.0, "SVD: s not positive");

        let utu = svd.u.transpose().dot(&svd.u);
        let eye3 = Matrix::eye(3);
        assert_matrix_approx(&utu, &eye3, 1e-4, "SVD: UtU = I");

        let vtv = svd.vt.dot(&svd.vt.transpose());
        let eye2 = Matrix::eye(2);
        assert_matrix_approx(&vtv, &eye2, 1e-4, "SVD: Vt*Vt^T = I");

        println!("  ✓ SVD");
    }

    {
        let a = Matrix::new(vec![4.0, 2.0, 2.0, 3.0], [2, 2]);
        let l = a.cholesky().expect("cholesky failed");

        let reconstructed = l.dot(&l.transpose());
        assert_matrix_approx(&reconstructed, &a, 1e-4, "Cholesky: L*Lt = A");

        assert_approx(unsafe { *l.get_unchecked(0, 1) }, 0.0, 1e-5, "Cholesky: upper zero");
        println!("  ✓ Cholesky");
    }
}

fn test_math() {
    println!("Testing math operations...");

    {
        let a = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
        let at = a.transpose();
        assert_approx(unsafe { *at.get_unchecked(0, 1) }, 3.0, 1e-5, "transpose (0,1)");
        assert_approx(unsafe { *at.get_unchecked(1, 0) }, 2.0, 1e-5, "transpose (1,0)");
        println!("  ✓ Transpose");
    }

    {
        let a = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
        assert_approx(a.trace(), 5.0, 1e-5, "trace");
        println!("  ✓ Trace");
    }

    {
        let a = Matrix::new(vec![3.0, 8.0, 4.0, 6.0], [2, 2]);
        let det = a.det().expect("det failed");
        assert_approx(det, -14.0, 1e-3, "det");
        println!("  ✓ Det");
    }

    {
        let a = Matrix::new(vec![4.0, 7.0, 2.0, 6.0], [2, 2]);
        let inv = a.inv().expect("inv failed");
        let eye = a.dot(&inv);
        assert_matrix_approx(&eye, &Matrix::eye(2), 1e-4, "inv: A*A^-1 = I");
        println!("  ✓ Inv");
    }

    { 
        let rank_deficient = Matrix::new(vec![1.0, 2.0, 2.0, 4.0], [2, 2]);
        let svd = rank_deficient.svd().unwrap();
        println!("singular values: {:?}", svd.s.data);
        println!("threshold: {}", 1e-10f32 * svd.s.data[0]);
        println!("rank: {}", rank_deficient.rank());
        let full_rank = Matrix::new(vec![1.0, 0.0, 0.0, 1.0], [2, 2]);
        assert_eq!(full_rank.rank(), 2, "rank full");

        let rank_deficient = Matrix::new(vec![1.0, 2.0, 2.0, 4.0], [2, 2]);
        assert_eq!(rank_deficient.rank(), 1, "rank deficient");
        println!("  ✓ Rank");
    }

    {
        let a = Matrix::new(vec![2.0, 1.0, 1.0, 3.0], [2, 2]);
        let b = Vector::from(vec![5.0, 10.0]);
        let x = a.solve(&b).expect("solve failed");
        assert_approx(unsafe { *x.get_unchecked(0) }, 1.0, 1e-4, "solve x");
        assert_approx(unsafe { *x.get_unchecked(1) }, 3.0, 1e-4, "solve y");
        println!("  ✓ Solve");
    }

    {
        let x = Matrix::new(vec![
            1.0, 1.0,
            1.0, 2.0,
            1.0, 3.0,
            1.0, 4.0,
            1.0, 5.0,
        ], [5, 2]);
        let y = Vector::from(vec![3.0, 5.0, 7.0, 9.0, 11.0]);
        let weights = x.pseudo_inv().expect("pseudo_inv failed").dot(&y);
        assert_approx(unsafe { *weights.get_unchecked(0) }, 1.0, 1e-3, "pseudo_inv bias");
        assert_approx(unsafe { *weights.get_unchecked(1) }, 2.0, 1e-3, "pseudo_inv slope");
        println!("  ✓ Pseudo Inverse");
    }
}