// ECSLRx_compute.cpp
// Optimized fork of ECSLRmulti_compute.cpp
//
// Key optimizations vs original:
//   1 (admm_flasso): Replace explicit ADMM_aux matrix rebuild on every
//          tau update with a scale-vector approach: keep Qt pre-computed,
//          recompute only the p-length scale vector when tau changes.
//          beta_kappa = Q * (scale .* (Qt * rhs))  -- O(p) update instead of O(p^2).
//
//   2 (ECSLR_Compute_Coef_combine): Incremental B_norm2 update.
//          Only recompute column m after betas.col(m) changes, not the full J*M matrix.
//
//   3 (ECSLR_Compute_Coef_combine): Accept pre-computed Q and eigval from R level
//          to skip redundant compute_block_eigen calls across CV iterations.

#include <omp.h>
#include <RcppEigen.h>
#include <cstdlib>
#include <cmath>
#include <string>

using namespace Rcpp;

bool ecslr_verbose_enabled() {
  const char* value = std::getenv("ECSLR_VERBOSE");
  if (value == nullptr) return false;
  std::string flag(value);
  return flag == "1" || flag == "true" || flag == "TRUE" || flag == "yes" || flag == "YES";
}

// [[Rcpp::export]]
Eigen::MatrixXd Matrix_Mult(const Eigen::Map<Eigen::MatrixXd> A,
                 const Eigen::Map<Eigen::MatrixXd> B,
                 int n_cores=4){
  Eigen::setNbThreads(n_cores);
  Eigen::MatrixXd C = A * B;
  return C;
}

// [[Rcpp::export]]
Eigen::MatrixXd Matrix_crossprod(const Eigen::Map<Eigen::MatrixXd> A,
                      const Eigen::Map<Eigen::MatrixXd> B,
                      int n_cores=4){
  Eigen::setNbThreads(n_cores);
  Eigen::MatrixXd C = A.transpose() * B;
  return C;
}

// [[Rcpp::export]]
Eigen::VectorXd Soft(const Eigen::VectorXd& z, const Eigen::VectorXd& gamma) {
  return z.array().sign() * (z.array().abs() - gamma.array()).cwiseMax(0);
}

// [[Rcpp::export]]
Eigen::VectorXd group_norm(const Eigen::VectorXd& s,
                           const Eigen::Map<Eigen::SparseMatrix<double>> Group_Matrix){
  return (Group_Matrix * s.array().square().matrix()).cwiseSqrt();
}

// [[Rcpp::export]]
Eigen::VectorXd group_soft_thresh(const Eigen::VectorXd& vv,
                                  const Eigen::VectorXd& ue,
                                  const Eigen::Map<Eigen::SparseMatrix<double>> Group_Matrix) {
  Eigen::VectorXd vv_norm2 = group_norm(vv, Group_Matrix);
  return (Group_Matrix.transpose() * ((1.0 - ue.array()/vv_norm2.array()).cwiseMax(0)).matrix()).array() * vv.array();
}

// [[Rcpp::export]]
Eigen::VectorXd sigmoid(const Eigen::VectorXd& z) {
  return static_cast<double>(1.0) / (1.0 + (-z).array().exp());
}

// [[Rcpp::export]]
double Compute_penalty_sparsity_Lasso(const int& m,
                                      const Eigen::MatrixXd& betas,
                                      const Eigen::Map<Eigen::VectorXd> w_lasso,
                                      const double& lambda_lasso) {
  return lambda_lasso * (w_lasso.array() * betas.col(m).array()).abs().sum();
}

// [[Rcpp::export]]
double Compute_penalty_sparsity_groupLasso(const int& m,
                                           const Eigen::MatrixXd& betas,
                                           const Eigen::Map<Eigen::SparseMatrix<double>> Group_Matrix,
                                           const Eigen::Map<Eigen::VectorXd> w_group,
                                           const double& lambda_group) {
  return lambda_group * (w_group.transpose() * (Group_Matrix * betas.col(m).array().square().matrix()).cwiseSqrt().matrix()).value();
}

// [[Rcpp::export]]
double Compute_penalty_fused(const int& m,
                             const Eigen::MatrixXd& betas,
                             const Eigen::Map<Eigen::SparseMatrix<double>> fused_penmat,
                             const double& lambda_fused) {
  return lambda_fused * ((fused_penmat * betas.col(m)).array().abs().sum());
}

// [[Rcpp::export]]
double Compute_penalty_diversity_group(const int& m, const Eigen::MatrixXd& betas,
                                       const Eigen::Map<Eigen::SparseMatrix<double>> Group_Matrix,
                                       const Eigen::Map<Eigen::VectorXd> w_group,
                                       const double& lambda_diversity) {
  Eigen::MatrixXd beta_norm2 = (Group_Matrix * betas.array().square().matrix()).cwiseSqrt();
  return lambda_diversity * 0.5 * ((w_group.array().square() * (beta_norm2.col(m).array() * ((beta_norm2.rowwise().sum() - beta_norm2.col(m)).array()))).sum());
}

// [[Rcpp::export]]
double Compute_AEC(const Eigen::VectorXd& scores,
                   const Eigen::Map<Eigen::VectorXd> y,
                   const Eigen::Map<Eigen::MatrixXd> cost_matrix) {
  return (y.array() * (scores.array() * cost_matrix.col(3).array() + (1 - scores.array()) * cost_matrix.col(2).array()) +
          (1 - y.array()) * (scores.array() * cost_matrix.col(1).array() + (1 - scores.array()) * cost_matrix.col(0).array())).mean();
}

// [[Rcpp::export]]
double Compute_objective_multi(const int& m,
                               const Eigen::Map<Eigen::MatrixXd> x_std,
                               const Eigen::Map<Eigen::VectorXd> y,
                               const Eigen::Map<Eigen::MatrixXd> cost_matrix,
                               const double& lambda_lasso,
                               const double& lambda_group,
                               const double& lambda_fused,
                               const double& lambda_diversity,
                               const Eigen::Map<Eigen::VectorXd> w_lasso,
                               const Eigen::Map<Eigen::VectorXd> w_group,
                               const Eigen::Map<Eigen::SparseMatrix<double>> Group_Matrix,
                               const Eigen::Map<Eigen::SparseMatrix<double>> fused_penmat,
                               const Eigen::MatrixXd& betas,
                               const Eigen::RowVectorXd& intercept) {
  Eigen::VectorXd z = (x_std * betas.col(m)).array() + intercept[m];
  Eigen::VectorXd scores = sigmoid(z);
  return (Compute_AEC(scores, y, cost_matrix)+
          Compute_penalty_sparsity_Lasso(m, betas, w_lasso, lambda_lasso)+
          Compute_penalty_sparsity_groupLasso(m, betas, Group_Matrix, w_group, lambda_group)+
          Compute_penalty_fused(m, betas, fused_penmat, lambda_fused)+
          Compute_penalty_diversity_group(m, betas, Group_Matrix, w_group, lambda_diversity));
}

// [[Rcpp::export]]
double Compute_objective_ensemble_multi(const int& M,
                                        const Eigen::Map<Eigen::MatrixXd> x_std,
                                        const Eigen::Map<Eigen::VectorXd> y,
                                        const Eigen::Map<Eigen::MatrixXd> cost_matrix,
                                        const double& lambda_lasso,
                                        const double& lambda_group,
                                        const double& lambda_fused,
                                        const double& lambda_diversity,
                                        const Eigen::Map<Eigen::VectorXd> w_lasso,
                                        const Eigen::Map<Eigen::VectorXd> w_group,
                                        const Eigen::Map<Eigen::SparseMatrix<double>> Group_Matrix,
                                        const Eigen::Map<Eigen::SparseMatrix<double>> fused_penmat,
                                        const Eigen::MatrixXd& betas,
                                        const Eigen::RowVectorXd& intercept) {
  Eigen::VectorXd obj_M = Eigen::VectorXd::Zero(M);
  #pragma omp parallel for
  for (int m = 0; m < M; ++m) {
    Eigen::VectorXd z = (x_std * betas.col(m)).array() + intercept[m];
    Eigen::VectorXd scores = sigmoid(z);
    obj_M[m] =  Compute_AEC(scores, y, cost_matrix)+
      Compute_penalty_sparsity_Lasso(m, betas, w_lasso, lambda_lasso)+
      Compute_penalty_sparsity_groupLasso(m, betas, Group_Matrix, w_group, lambda_group)+
      Compute_penalty_fused(m, betas, fused_penmat, lambda_fused)+
      Compute_penalty_diversity_group(m, betas, Group_Matrix, w_group, lambda_diversity)/2;
  }
  return obj_M.sum();
}

// =========================================================================
// OPT-1: admm_flasso -- scale-vector trick (no explicit ADMM_aux matrix)
//
// Original stores ADMM_aux = Q * diag(scale) * Qt (p×p matrix) and rebuilds
// it on every tau change. This costs O(p^2) memory and O(p^2) multiply each
// time tau changes.
//
// New approach: store Qt (pre-computed once), keep scale as a p-length
// vector, and compute beta_kappa as:
//   Qt_rhs = Qt * rhs          -- O(p * sparsity)
//   beta_kappa = Q * (scale .* Qt_rhs)  -- O(p * sparsity)
// tau update only refreshes scale (O(p)), no matrix rebuild.
// =========================================================================
// [[Rcpp::export]]
Eigen::VectorXd admm_flasso(const Eigen::VectorXd& vv,
                            const Eigen::Map<Eigen::SparseMatrix<double>> Group_Matrix,
                            const Eigen::Map<Eigen::SparseMatrix<double>> Adjust_Matrix_Fused,
                            const Eigen::VectorXd& lambda_3,
                            const Eigen::Map<Eigen::SparseMatrix<double>> fused_penmat,
                            const Eigen::SparseMatrix<double>& Q,
                            const Eigen::VectorXd& eigval,
                            const int& max_iter_admm,
                            const Eigen::VectorXd& betas_old,
                            const Eigen::VectorXd& numfused,
                            const Eigen::VectorXd& df,
                            int& iter_count){

  Eigen::SparseMatrix<double> penmat_t = fused_penmat.transpose();
  int J = Group_Matrix.rows();
  int p = Group_Matrix.cols();
  int Nfused = Adjust_Matrix_Fused.cols();

  // Initialize values
  Eigen::VectorXd beta_kappa = Eigen::VectorXd::Zero(p);
  Eigen::VectorXd xhat = Eigen::VectorXd::Zero(Nfused);
  Eigen::VectorXd z_old = Eigen::VectorXd::Zero(Nfused);
  // Use starting value for beta
  Eigen::VectorXd z_new = fused_penmat * betas_old;
  // Starting value for q_kappa is zero vector
  Eigen::VectorXd q_kappa = Eigen::VectorXd::Zero(Nfused);
  Eigen::VectorXd pbk(Nfused);

  Eigen::VectorXd eps_pri(J);
  Eigen::VectorXd eps_dual(J);
  Eigen::VectorXd r_norm(J);
  Eigen::VectorXd s_norm(J);

  Eigen::VectorXd tau = Eigen::VectorXd::Ones(J);
  Eigen::VectorXd tau_old = tau;

  // Relative tolerance
  double eps_rel = std::pow(10, -10.0);
  // Absolute tolerance
  double eps_abs = std::pow(10, -12.0);
  double xi = 1.5;

  // OPT-1 (fixed): Store Qt as sparse matrix for memory-efficient and fast block-diagonal matvec.
  Eigen::SparseMatrix<double> Qt = Q.transpose();



  // scale[i] = 1 / (1 + tau_p[i] * eigval[i])
  // where tau_p = Group_Matrix.T * tau  (broadcasts group tau to predictor level)
  auto compute_scale = [&]() -> Eigen::VectorXd {
    return (1.0 / (1.0 + (Group_Matrix.transpose() * tau).array() * eigval.array())).matrix();
  };

  Eigen::VectorXd scale = compute_scale();  // p-length vector

  double mu = 10;
  double eta = 2;

  Eigen::VectorXd multi = Eigen::VectorXd::Ones(J);
  Eigen::VectorXd eta_vec = Eigen::VectorXd::Constant(J, eta);
  Eigen::VectorXd eta_inv_vec = Eigen::VectorXd::Constant(J, static_cast<double>(1.0)/eta);

  Eigen::VectorXd r_over_eps(J);
  Eigen::VectorXd s_over_eps(J);

  int iter = 1;
  for(iter = 1; iter <= max_iter_admm; ++iter){

    z_old = z_new;


    // beta_kappa = ADMM_aux * (vv + ((Group_Matrix.transpose() * tau).array() * (penmat_t * (z_old - q_kappa)).array()).matrix());


    // Update beta_kappa using scale-vector trick (OPT-1, sparse-only):
    //   rhs = vv + tau_p .* (penmat_t * (z_old - q_kappa))
    //   beta_kappa = Q * (scale .* (Qt * rhs))
    Eigen::VectorXd tau_p = (Group_Matrix.transpose() * tau);
    Eigen::VectorXd rhs = vv + (tau_p.array() * (penmat_t * (z_old - q_kappa)).array()).matrix();
    Eigen::VectorXd Qt_rhs = Qt * rhs;     // sparse matvec: p-vector
    beta_kappa = Q * (scale.array() * Qt_rhs.array()).matrix();  // sparse matvec: p-vector

    // Relaxation
    xhat = xi * (fused_penmat * beta_kappa) + (1 - xi) * z_old;

    // Update z
    z_new = Soft(xhat + q_kappa, Adjust_Matrix_Fused.transpose() * (lambda_3.array() / tau.array()).matrix());

    // Update q_kappa

    // Rcout << "iter=" << iter << std::endl;
    // Rcout << "beta_kappa=" << beta_kappa << std::endl;

    // Rcout << "xhat=" << xhat << std::endl;
    // Rcout << "z_new=" << z_new << std::endl;
    // Rcout << "tau=" << tau << std::endl;

    q_kappa = q_kappa + xhat - z_new;

    pbk << fused_penmat * beta_kappa;

    // Convergence checks
    eps_pri = numfused.cwiseSqrt() * eps_abs + eps_rel * (group_norm(pbk, Adjust_Matrix_Fused).array().max(group_norm(z_new,Adjust_Matrix_Fused).array())).matrix();
    // Tolerance for dual feasibility condition
    eps_dual = df.cwiseSqrt() * eps_abs + eps_rel * (tau.array() * group_norm(penmat_t*q_kappa, Group_Matrix).array()).matrix();
    // Norm of primal residuals
    r_norm = group_norm(pbk - z_new, Adjust_Matrix_Fused);
    // Norm of dual residuals
    s_norm = group_norm(((Group_Matrix.transpose() * (-tau)).array() * (penmat_t * (z_new - z_old)).array()).matrix(), Group_Matrix);

    // Rcout << "eps_pri=" << eps_pri << std::endl;
    // Rcout << "eps_dual=" << eps_dual << std::endl;
    // Rcout << "r_norm=" << r_norm << std::endl;
    // Rcout << "s_norm=" << s_norm << std::endl;


    if((r_norm.array() <= eps_pri.array()).all() && (s_norm.array() <= eps_dual.array()).all()){
      break;
    }else{
      r_over_eps = r_norm.array() / eps_pri.array();
      s_over_eps = s_norm.array() / eps_dual.array();

      multi = (r_over_eps.array() >= mu * s_over_eps.array()).select(eta_vec,
               (s_over_eps.array() >= mu * r_over_eps.array()).select(eta_inv_vec, 1.0));

      tau_old = tau;
      // Update tau
      tau = (multi.array() * tau_old.array()).matrix();
      // ADMM_aux = Q * ((static_cast<double>(1.0) / (1.0 + (Group_Matrix.transpose() * tau).array() * eigval.array())).matrix().asDiagonal()) * Qt;
      // OPT-1: Only update scale vector (O(p)), no matrix rebuild
      scale = compute_scale();

      // Update q_kappa
      q_kappa = q_kappa.array() * (Adjust_Matrix_Fused.transpose() * (tau_old.array() / tau.array()).matrix()).array();
    }
  }
  iter_count = (iter > max_iter_admm) ? max_iter_admm : iter;
  return beta_kappa;
}

// =========================================================================
// compute_block_eigen (exported to R)
//
// Computes the block-wise eigensystem for the adaptive tau ADMM solver.
// Group_Matrix and Adjust_Matrix_Fused are constructed in contiguous group
// order, so the block sizes are just their row sums.
// =========================================================================
// [[Rcpp::export]]
Rcpp::List compute_block_eigen(
    const Eigen::Map<Eigen::SparseMatrix<double>> Group_Matrix,
    const Eigen::Map<Eigen::SparseMatrix<double>> Adjust_Matrix_Fused,
    const Eigen::Map<Eigen::SparseMatrix<double>> fused_penmat)
{
  int J = Group_Matrix.rows();
  int p = Group_Matrix.cols();
  int Nfused = Adjust_Matrix_Fused.cols();

  // if(fused_penmat.cols() != p){
  //   Rcpp::stop("ncol(Group_Matrix) must equal ncol(fused_penmat)");
  // }
  // if(fused_penmat.rows() != Nfused){
  //   Rcpp::stop("ncol(Adjust_Matrix_Fused) must equal nrow(fused_penmat)");
  // }

  Eigen::VectorXi p_sizes = (Group_Matrix * Eigen::VectorXd::Ones(p)).cast<int>();
  Eigen::VectorXi nf_sizes = (Adjust_Matrix_Fused * Eigen::VectorXd::Ones(Nfused)).cast<int>();

  Eigen::VectorXd eigval = Eigen::VectorXd::Zero(p);
  Eigen::MatrixXd Q_dense = Eigen::MatrixXd::Zero(p, p);

  int col_start = 0;
  int row_start = 0;
  for(int j = 0; j < J; ++j){
    int pj = p_sizes[j];
    int nfj = nf_sizes[j];

    if(nfj == 0){
      Q_dense.block(col_start, col_start, pj, pj).setIdentity();
      col_start += pj;
      row_start += nfj;
      continue;
    }

    auto Gj = fused_penmat.block(row_start, col_start, nfj, pj);
    Eigen::MatrixXd gram = Eigen::MatrixXd(Gj.transpose() * Gj);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(gram);
    Eigen::MatrixXd Qj = eig.eigenvectors();
    Eigen::VectorXd Mj = eig.eigenvalues();

    eigval.segment(col_start, pj) = Mj;
    Q_dense.block(col_start, col_start, pj, pj) = Qj;

    col_start += pj;
    row_start += nfj;
  }

  Eigen::SparseMatrix<double> Q = Q_dense.sparseView(1e-14);
  Q.makeCompressed();

  return Rcpp::List::create(Rcpp::Named("Q") = Q,
                            Rcpp::Named("eigval") = eigval);
}


// [[Rcpp::export]]
Eigen::VectorXd update_betas(const Eigen::VectorXd& vv,
                             const Eigen::VectorXd& betas_old,
                             const Eigen::Map<Eigen::SparseMatrix<double>> Group_Matrix,
                             const Eigen::Map<Eigen::SparseMatrix<double>> Adjust_Matrix_Fused,
                             const Eigen::Map<Eigen::SparseMatrix<double>> fused_penmat,
                             const Eigen::SparseMatrix<double>& Q,
                             const Eigen::VectorXd& eigval,
                             const Eigen::VectorXd& numfused,
                             const Eigen::VectorXd& df,
                             const int& max_iter_admm,
                             const double& lambda_lasso,
                             const Eigen::Map<Eigen::VectorXd> w_lasso,
                             const double& lambda_fused,
                             const Eigen::VectorXd& group_soft_penalty,
                             const Eigen::VectorXd& eta_m,
                             const bool& exist_grouplasso,
                             Eigen::VectorXd& lambda_1w,
                             Eigen::VectorXd& lambda_2,
                             Eigen::VectorXd& lambda_3){

  Eigen::VectorXd beta_0_0_flasso = vv;
  if(lambda_fused!=0){
    lambda_3 = lambda_fused * eta_m;
    int dummy_iter = 0;
    beta_0_0_flasso << admm_flasso(vv,
                                   Group_Matrix,
                                   Adjust_Matrix_Fused,
                                   lambda_3,
                                   fused_penmat,
                                   Q,
                                   eigval,
                                   max_iter_admm,
                                   betas_old,
                                   numfused,
                                   df,
                                   dummy_iter);
  }

  Eigen::VectorXd beta_lasso_0_flasso = beta_0_0_flasso;
  if(lambda_lasso != 0){
    lambda_1w << (Group_Matrix.transpose() * (lambda_lasso * eta_m)).array() * w_lasso.array();
    beta_lasso_0_flasso << Soft(beta_0_0_flasso, lambda_1w);
  }

  Eigen::VectorXd beta_lasso_glasso_flasso = beta_lasso_0_flasso;
  if(exist_grouplasso){
    lambda_2 << group_soft_penalty.array() * eta_m.array();
    beta_lasso_glasso_flasso = group_soft_thresh(beta_lasso_0_flasso, lambda_2, Group_Matrix);
  }

  return beta_lasso_glasso_flasso;
}


// [[Rcpp::export]]
Rcpp::List update_betas_with_iter(const Eigen::VectorXd& vv,
                                  const Eigen::VectorXd& betas_old,
                                  const Eigen::Map<Eigen::SparseMatrix<double>> Group_Matrix,
                                  const Eigen::Map<Eigen::SparseMatrix<double>> Adjust_Matrix_Fused,
                                  const Eigen::Map<Eigen::SparseMatrix<double>> fused_penmat,
                                  const Eigen::SparseMatrix<double>& Q,
                                  const Eigen::VectorXd& eigval,
                                  const Eigen::VectorXd& numfused,
                                  const Eigen::VectorXd& df,
                                  const int& max_iter_admm,
                                  const double& lambda_lasso,
                                  const Eigen::Map<Eigen::VectorXd> w_lasso,
                                  const double& lambda_fused,
                                  const Eigen::VectorXd& group_soft_penalty,
                                  const Eigen::VectorXd& eta_m,
                                  const bool& exist_grouplasso) {

  Eigen::VectorXd beta_0_0_flasso = vv;
  int iter_count = 0;

  Eigen::VectorXd lambda_1w = Eigen::VectorXd::Zero(vv.size());
  Eigen::VectorXd lambda_2 = Eigen::VectorXd::Zero(group_soft_penalty.size());
  Eigen::VectorXd lambda_3 = Eigen::VectorXd::Zero(eta_m.size());

  if (lambda_fused != 0) {
    lambda_3 = lambda_fused * eta_m;
    beta_0_0_flasso = admm_flasso(vv,
                                  Group_Matrix,
                                  Adjust_Matrix_Fused,
                                  lambda_3,
                                  fused_penmat,
                                  Q,
                                  eigval,
                                  max_iter_admm,
                                  betas_old,
                                  numfused,
                                  df,
                                  iter_count);
  }

  Eigen::VectorXd beta_lasso_0_flasso = beta_0_0_flasso;
  if (lambda_lasso != 0) {
    lambda_1w = (Group_Matrix.transpose() * (lambda_lasso * eta_m)).array() * w_lasso.array();
    beta_lasso_0_flasso = Soft(beta_0_0_flasso, lambda_1w);
  }

  Eigen::VectorXd beta_lasso_glasso_flasso = beta_lasso_0_flasso;
  if (exist_grouplasso) {
    lambda_2 = group_soft_penalty.array() * eta_m.array();
    beta_lasso_glasso_flasso = group_soft_thresh(beta_lasso_0_flasso, lambda_2, Group_Matrix);
  }

  return Rcpp::List::create(Rcpp::Named("beta") = beta_lasso_glasso_flasso,
                            Rcpp::Named("iter") = iter_count);
}


// =========================================================================
// OPT-2 + OPT-3: ECSLR_Compute_Coef_combine with:
//   - OPT-2: Incremental B_norm2 -- only update col(m) after betas.col(m) changes
//   - OPT-3: Accept pre-computed Q/eigval via eigen_cache_ (Nullable List)
//            to skip compute_block_eigen calls when called from CV loops.
// =========================================================================
// [[Rcpp::export]]
List ECSLR_Compute_Coef_combine(const Eigen::Map<Eigen::MatrixXd> x,
                                const Eigen::Map<Eigen::VectorXd> y,
                                const Eigen::Map<Eigen::MatrixXd> cost_matrix,
                                const int& M,
                                const bool& include_intercept,
                                const Eigen::Map<Eigen::VectorXd> w_lasso,
                                const Eigen::Map<Eigen::SparseMatrix<double>> Group_Matrix,
                                const Eigen::Map<Eigen::VectorXd> w_group,
                                const Eigen::Map<Eigen::SparseMatrix<double>> Adjust_Matrix_Fused,
                                const Eigen::Map<Eigen::SparseMatrix<double>> fused_penmat,
                                const double& alpha,
                                const double& gamma,
                                const double& lambda_sparsity,
                                const double& lambda_diversity,
                                const double& tolerance,
                                const double& obj_tol_rel,
                                const int& max_iter,
                                const int& max_iter_admm,
                                const std::string& model,
                                const Rcpp::Nullable<Rcpp::List>& x0_ = R_NilValue,
                                const Rcpp::Nullable<Rcpp::List>& eigen_cache_ = R_NilValue
) {
  // OPT-3: eigen_cache_ optionally holds pre-computed list(Q, eigval) from R level
  const int n = x.rows();
  const int p = x.cols();
  const int J = Group_Matrix.rows();
  const bool verbose = ecslr_verbose_enabled();
  const Eigen::VectorXd df = Group_Matrix * Eigen::VectorXd::Ones(p);

  const double CCSA_RHOMIN = 1e-5;
  const double OBJ_TOLERANCE_REL = obj_tol_rel;
  const double BETAS_TOLERANCE_ABS = tolerance;
  const int maxK = 100;
  const int maxL = 100;

  Eigen::MatrixXd betas(p,M);
  Eigen::RowVectorXd intercept(M);

  if (x0_.isNotNull()) {
    List x0(x0_);
    betas = x0["betas"];
    intercept = x0["intercept"];
  }else{
    betas = Eigen::MatrixXd::Zero(p, M);
    intercept = Eigen::RowVectorXd::Zero(1, M);
  }

  if (verbose) Rcout <<"OBJ_TOLERANCE_REL="<<OBJ_TOLERANCE_REL<<", intercept=" << intercept.head(M) << std::endl;

  const Eigen::RowVectorXd mu_x = x.colwise().mean();
  const Eigen::RowVectorXd sd_x = ((x.rowwise() - mu_x).array().square().matrix().colwise().sum()/(n-1)).array().sqrt();
  Eigen::MatrixXd x_std = (x.rowwise() - mu_x).array().rowwise() / sd_x.array();
  x_std = x_std.array().isNaN().select(0.0, x_std.array());
  // define some map object.
  Eigen::Map<Eigen::MatrixXd> x_std_map(x_std.data(), x_std.rows(), x_std.cols());

  // initialize the parameter.
  Eigen::MatrixXd new_betas = betas;
  Eigen::RowVectorXd new_intercept = intercept;
  Eigen::MatrixXd betas_prev = betas;
  Eigen::RowVectorXd intercept_prev = intercept;
  Eigen::MatrixXd betas_prevprev = betas;
  Eigen::RowVectorXd intercept_prevprev = intercept;

  const Eigen::VectorXd b = y.array() * (cost_matrix.col(3) - cost_matrix.col(2)).array() +
    (1 - y.array()) * (cost_matrix.col(1) - cost_matrix.col(0)).array();

  Eigen::RowVectorXd rho = Eigen::RowVectorXd::Ones(M);

  Eigen::VectorXd betas_mean(p);
  Eigen::RowVectorXd sigm0 = Eigen::RowVectorXd::Ones(M);
  Eigen::MatrixXd sigm = Eigen::MatrixXd::Ones(J, M);

  double obj_full=10000;
  double obj_full_new=obj_full;
  double dintercept2=0;
  double gam0=0;
  Eigen::VectorXd dbetas2(p);
  Eigen::VectorXd gam(p);

  Eigen::VectorXd z(n);
  Eigen::VectorXd expected_val_g(n);
  Eigen::VectorXd d(n);
  Eigen::VectorXd x_std_d(p);

  double aec_g = 0;
  Eigen::VectorXd new_z(n);
  Eigen::VectorXd new_expect_val(n);
  Eigen::VectorXd eta_m_vec(J);

  const double lambda_lasso = lambda_sparsity * alpha * gamma;
  const double lambda_group = lambda_sparsity * (1.0-alpha) * gamma;
  const double lambda_fused = lambda_sparsity * (1.0-gamma);

  if (verbose) Rcout << "lambda_lasso=" << lambda_lasso<< ", lambda_group=" << lambda_group<< ", lambda_fused=" << lambda_fused << std::endl;

  const bool exist_grouplasso = (lambda_group != 0) || (lambda_diversity != 0);

  Eigen::VectorXd lambda_1w(p);
  Eigen::VectorXd lambda_2(J);
  Eigen::VectorXd lambda_3(J);
  Eigen::VectorXd group_soft_penalty(J);
  Eigen::VectorXd vv(p);
  Eigen::VectorXd vv_norm2(J);

  double grad_g=0;
  double w=0;
  double quad_g=0;
  double approxi_value=0;
  double actual_value=0;
  double delta=0;
  double max_diff=10000;
  double obj_rel_diff=10000;

  int r=0;

  Eigen::VectorXd eigval(p);
  Eigen::SparseMatrix<double> Q(p,p);
  Eigen::VectorXd numfused(J);

  // OPT-3: eigen_cache_ is prepared once at R level and consumed here.
  if(lambda_fused!=0){
    const int Nfused = Adjust_Matrix_Fused.cols();
    numfused = Adjust_Matrix_Fused * Eigen::VectorXd::Ones(Nfused);
    Rcpp::List eigen_res(eigen_cache_);
    Q = Rcpp::as<Eigen::SparseMatrix<double>>(eigen_res["Q"]);
    eigval = Rcpp::as<Eigen::VectorXd>(eigen_res["eigval"]);
  }

  Eigen::VectorXd obj = Eigen::VectorXd::Constant(max_iter, -1);
  Eigen::VectorXd obj_M_vec = Eigen::VectorXd::Zero(M);
  Eigen::VectorXd diff_com = Eigen::VectorXd::Constant(max_iter, -1);
  double obj_final=10000;
  int cccount = 0;

  // OPT-2: Initialize B_norm2 once; update only col(m) incrementally
  Eigen::MatrixXd B_norm2 = (Group_Matrix * betas.array().square().matrix()).cwiseSqrt().array();

  // main loop
  for (r = 0; r < max_iter; ++r) {
    // Rcout << "r=" << r << std::endl;
    new_betas = betas;
    new_intercept = intercept;
    betas_mean = betas.rowwise().mean();

    sigm0 = Eigen::RowVectorXd::Ones(M);
    sigm = Eigen::MatrixXd::Ones(J, M);

    for (int m = 0; m < M; ++m) {

      obj_full_new = Compute_objective_multi(m, x_std_map, y, cost_matrix,
                                             lambda_lasso, lambda_group, lambda_fused, lambda_diversity,
                                             w_lasso, w_group, Group_Matrix, fused_penmat,
                                             betas, intercept);

      for (int k = 1; k <= maxK; ++k) {
        // Rcout << "m:" << m << ", k:" << k << std::endl;

        obj_full = obj_full_new;

        if(k==1){
          rho[m] = 1.0;
        }else{
          rho[m] = std::max(0.1*rho[m],CCSA_RHOMIN);
        }
        // Rcout << "rho[m]=" << rho[m] << std::endl;

        if(k >=3){
          dintercept2 = (intercept[m] - intercept_prev[m]) * (intercept_prev[m] - intercept_prevprev[m]);
          gam0 = (dintercept2 < 0) ? 0.7 : ((dintercept2 > 0) ? 1.2 : 1.0);
          sigm0[m] = gam0 * sigm0[m];
          dbetas2 = Group_Matrix * ((betas.col(m) - betas_prev.col(m)).array() * (betas_prev.col(m) - betas_prevprev.col(m)).array()).matrix();
          // dbetas2 << -1, 0, 1;
          gam = dbetas2.array().unaryExpr([](double val) {
            return (val < 0) ? 0.7 : ((val > 0) ? 1.2 : 1.0);});
          sigm.col(m) = gam.array() * sigm.col(m).array();
        }

        z << (x_std * betas.col(m)).array() + intercept[m];
        expected_val_g << sigmoid(z);
        aec_g = Compute_AEC(expected_val_g, y, cost_matrix);
        d = b.array() * expected_val_g.array() * (1-expected_val_g.array());

        // OPT-2: Use incrementally-maintained B_norm2; compute col(m) for current betas
        B_norm2.col(m) = (Group_Matrix * betas.col(m).array().square().matrix()).cwiseSqrt();
        group_soft_penalty << lambda_group * w_group + (lambda_diversity * 0.5) * (B_norm2.rowwise().sum() - B_norm2.col(m));

        x_std_d << (static_cast<double>(1.0) / n) * (x_std.transpose() * d);

        for (int l = 1; l <= maxL; ++l) {

          cccount++;
          new_intercept[m] = include_intercept ? (intercept[m] - std::pow(sigm0[m],2)/rho[m] * d.mean()):(mu_x * (betas.col(m).array()/sd_x.transpose().array()).matrix());

          eta_m_vec << sigm.col(m).array().square()/rho[m];
          vv << betas.col(m).array() - (Group_Matrix.transpose() * eta_m_vec).array() * x_std_d.array();

          new_betas.col(m) = update_betas(vv, betas.col(m), Group_Matrix,
                        Adjust_Matrix_Fused, fused_penmat, Q, eigval, numfused, df, max_iter_admm,
                        lambda_lasso, w_lasso, lambda_fused, group_soft_penalty, eta_m_vec, exist_grouplasso,
                        lambda_1w, lambda_2, lambda_3);

          grad_g = (x_std_d.transpose() * (new_betas.col(m) - betas.col(m))).value() + d.mean() * (new_intercept[m] - intercept[m]);
          w = 0.5 * ( std::pow((new_intercept[m] - intercept[m])/sigm0[m],2)
                        + ((new_betas.col(m) - betas.col(m)).array() / (Group_Matrix.transpose() * sigm.col(m)).array()).square().sum());
          quad_g = rho[m] * w;
          approxi_value = aec_g + grad_g + quad_g;

          new_z << (x_std * new_betas.col(m)).array() + new_intercept[m];
          // use "<<" instead of "=" so that the address of new_expect_val dose not change and the corresponding map can work.
          new_expect_val << sigmoid(new_z);
          actual_value = Compute_AEC(new_expect_val, y, cost_matrix);
          // Rcout << "l="<<  l <<  ", rho="<<  rho[m] <<  ", approxi_value=" << approxi_value << ", actual_value=" << actual_value << std::endl;

          // if approxi_value is upper bound, a better coefficients have been found
          if(approxi_value >= actual_value){
            break;
          }else{ // if approxi_value is not upper bound, change rho
            // update rho for iteration l+1
            delta = (actual_value-approxi_value)/w;
            rho[m] = std::min(10*rho[m],1.1*(rho[m] + delta));

            if(l==maxL){
              // 到达最大l的时候，如果还没发现合适解，则还是保持原来的解不动; 一般不会用到，大部分l在五次循环就找到合适解了。
              // return the updated beta to its original value
              new_intercept[m] = intercept[m];
              new_betas.col(m) = betas.col(m);
              break;
            }
          }
        } // end of inner iteration for l

        max_diff = std::max( (new_betas.col(m)-betas.col(m)).cwiseAbs().maxCoeff(), std::abs(new_intercept[m]-intercept[m]));

        obj_full_new = Compute_objective_multi(m, x_std_map, y, cost_matrix,
                                               lambda_lasso, lambda_group, lambda_fused, lambda_diversity,
                                               w_lasso, w_group, Group_Matrix, fused_penmat,
                                               new_betas, new_intercept);
        obj_rel_diff = (obj_full_new - obj_full)/obj_full;

        // Rcout << "m:" << m << ", k:" << k << ", objective on betas:" << obj_full << ", objective on newbetas:" << obj_full_new << std::endl;
        // Rcout << "m:" << m << ", k:" << k << ", max.betas.diff=" << max_diff << ", rel_diff=" << obj_rel_diff << std::endl;

        // didn't become better, use old solution, do not update betas and intercept
        if(obj_full_new > obj_full){

          new_betas.col(m) = betas.col(m);
          new_intercept[m] = intercept[m];
          //didn't become better, use old solution, do not update betas and intercept
          break;
        }

        // update betas and intercept m
        if(max_diff < BETAS_TOLERANCE_ABS || ( std::abs(obj_rel_diff) < OBJ_TOLERANCE_REL ) ){
          betas.col(m) = new_betas.col(m);
          intercept[m] = new_intercept[m];
          // OPT-2: Update B_norm2 col(m) after accepting new betas
          B_norm2.col(m) = (Group_Matrix * betas.col(m).array().square().matrix()).cwiseSqrt();
          break;
        }else{
          if(k >=2){
            betas_prevprev.col(m) = betas_prev.col(m);
            intercept_prevprev[m] = intercept_prev[m];
          }
          betas_prev.col(m) = betas.col(m);
          intercept_prev[m] = intercept[m];
          betas.col(m) = new_betas.col(m);
          intercept[m] = new_intercept[m];
          // OPT-2: Update B_norm2 col(m) incrementally
          B_norm2.col(m) = (Group_Matrix * betas.col(m).array().square().matrix()).cwiseSqrt();
        }

      } // end of outer iteration for k

    } // end of m

    obj[r] = Compute_objective_ensemble_multi(M, x_std_map, y, cost_matrix,
                                              lambda_lasso, lambda_group, lambda_fused, lambda_diversity,
                                              w_lasso, w_group, Group_Matrix, fused_penmat,
                                              betas, intercept);

    // diff_com = max(abs(betas-betas.r), abs(intercept-intercept.r))
    // Rcout << "betas_mean=" << betas_mean << std::endl;
    // Rcout << "betas.rowwise().mean()=" << betas.rowwise().mean() << std::endl;

    diff_com[r] = (betas.rowwise().mean()-betas_mean).array().square().maxCoeff();
    if (verbose) Rcout << "r=" << r << ", difference between betas and newbetas: " << diff_com[r] << ", obj=" << obj[r] << std::endl;

    if(diff_com[r] < tolerance){
      break;
    }
  }

  int num = std::min(r+1,max_iter);
  obj_final = obj[num-1];

  if (verbose) Rcout << "obj=" << obj.head(num).transpose()  << std::endl;
  if (verbose) Rcout << "diff_com=" << diff_com.head(num).transpose()  << std::endl;

  Eigen::MatrixXd betas_scaled = betas.array().colwise() / sd_x.transpose().array();
  betas_scaled = betas_scaled.unaryExpr([](double val) {
    return std::isnan(val) || std::isinf(val) ? 0.0 : val;
  });

  if (verbose) Rcout << "count=" << cccount << "\n" << std::endl;

  Eigen::RowVectorXd intercept_scaled = (include_intercept? 1: 0) * (intercept - mu_x * betas_scaled);
  Eigen::VectorXd betas_est = betas_scaled.rowwise().mean();
  double intercept_est = intercept_scaled.mean();

  return List::create(Named("betas_scaled") = betas_scaled,
                      Named("intercept_scaled") = intercept_scaled,
                      Named("betas_est") = betas_est,
                      Named("intercept_est") = intercept_est,
                      Named("alpha") = alpha,
                      Named("gamma") = gamma,
                      Named("lambda_sparsity") = lambda_sparsity,
                      Named("lambda_diversity") = lambda_diversity,
                      Named("M") = M,
                      Named("include_intercept") = include_intercept,
                      Named("model") = model,
                      Named("objective") = obj_final,
                      Named("coef_x_std") = List::create(
                        Named("betas") = betas,
                        Named("intercept") = intercept
                      )
  );
}
