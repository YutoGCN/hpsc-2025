#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>
#include <mpi.h>

using namespace std;
typedef vector<vector<float>> matrix;

int main(int argc, char** argv) {
  int nx = 500;
  int ny = 500;
  int nt = 500;
  int nit = 50;
  double dx = 2. / (nx - 1);
  double dy = 2. / (ny - 1);
  double dt = .01;
  double rho = 1.;
  double nu = .02;

  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int begin = rank * ny / size;
  int end = (rank + 1) * ny / size;

  matrix u(ny,vector<float>(nx));
  matrix v(ny,vector<float>(nx));
  matrix p(ny,vector<float>(nx));
  matrix b(ny,vector<float>(nx));
  matrix un(ny,vector<float>(nx));
  matrix vn(ny,vector<float>(nx));
  matrix pn(ny,vector<float>(nx));
  for (int j=0; j<ny; j++) {
    for (int i=0; i<nx; i++) {
      u[j][i] = 0;
      v[j][i] = 0;
      p[j][i] = 0;
      b[j][i] = 0;
    }
  }
  ofstream ufile("u.dat");
  ofstream vfile("v.dat");
  ofstream pfile("p.dat");
  for (int n=0; n<nt; n++) {
    for (int j = std::max(1, begin); j < std::min(ny - 1, end); j++) {
      for (int i = 1; i < nx - 1; i++) {
        b[j][i] = rho * (1.0 / dt * (
                      (u[j][i + 1] - u[j][i - 1]) / (2 * dx) +
                      (v[j + 1][i] - v[j - 1][i]) / (2 * dy)
                    )
                    - pow((u[j][i + 1] - u[j][i - 1]) / (2 * dx), 2)
                    - 2.0 * ((u[j + 1][i] - u[j - 1][i]) / (2 * dy)) * ((v[j][i + 1] - v[j][i - 1]) / (2 * dx))
                    - pow((v[j + 1][i] - v[j - 1][i]) / (2 * dy), 2)
                  );
      }
    }
    for (int it=0; it<nit; it++) {
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
	  pn[j][i] = p[j][i];
      for (int j = std::max(1, begin); j < std::min(ny - 1, end); j++) {
        for (int i = 1; i < nx - 1; i++) {
          p[j][i] = (dy * dy * (pn[j][i + 1] + pn[j][i - 1]) +
                     dx * dx * (pn[j + 1][i] + pn[j - 1][i]) -
                     b[j][i] * dx * dx * dy * dy)
                    / (2.0 * (dx * dx + dy * dy));
        }
      }
      // 境界条件
      for (int j=0; j<ny; j++) {
        p[j][nx-1] = p[j][nx-2]; // p[:, -1] = p[:, -2]
        p[j][0]    = p[j][1];    // p[:, 0] = p[:, 1]
      }
      for (int i=0; i<nx; i++) {
        p[0][i]    = p[1][i];    // p[0, :] = p[1, :]
        p[ny-1][i] = 0.0;        // p[-1, :] = 0
      }
    }
    for (int j=0; j<ny; j++) {
      for (int i=0; i<nx; i++) {
        un[j][i] = u[j][i];
	vn[j][i] = v[j][i];
      }
    }
    for (int j = std::max(1, begin); j < std::min(ny - 1, end); j++) {
      for (int i = 1; i < nx - 1; i++) {
        u[j][i] = un[j][i]
          - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1])
          - vn[j][i] * dt / dy * (un[j][i] - un[j - 1][i])
          - dt / (2.0 * rho * dx) * (p[j][i + 1] - p[j][i - 1])
          + nu * dt / (dx * dx) * (un[j][i + 1] - 2.0 * un[j][i] + un[j][i - 1])
          + nu * dt / (dy * dy) * (un[j + 1][i] - 2.0 * un[j][i] + un[j - 1][i]);

        v[j][i] = vn[j][i]
          - un[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1])
          - vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i])
          - dt / (2.0 * rho * dx) * (p[j + 1][i] - p[j - 1][i])
          + nu * dt / (dx * dx) * (vn[j][i + 1] - 2.0 * vn[j][i] + vn[j][i - 1])
          + nu * dt / (dy * dy) * (vn[j + 1][i] - 2.0 * vn[j][i] + vn[j - 1][i]);
      }
    }
    if (size > 1) {
      MPI_Status status;
      // u
      if (rank > 0) {
        MPI_Sendrecv(&u[begin][0], nx, MPI_FLOAT, rank - 1, 0,
                     &u[begin - 1][0], nx, MPI_FLOAT, rank - 1, 0,
                     MPI_COMM_WORLD, &status);
      }
      if (rank < size - 1) {
        MPI_Sendrecv(&u[end - 1][0], nx, MPI_FLOAT, rank + 1, 0,
                     &u[end][0], nx, MPI_FLOAT, rank + 1, 0,
                     MPI_COMM_WORLD, &status);
      }
      // v
      if (rank > 0) {
        MPI_Sendrecv(&v[begin][0], nx, MPI_FLOAT, rank - 1, 1,
                     &v[begin - 1][0], nx, MPI_FLOAT, rank - 1, 1,
                     MPI_COMM_WORLD, &status);
      }
      if (rank < size - 1) {
        MPI_Sendrecv(&v[end - 1][0], nx, MPI_FLOAT, rank + 1, 1,
                     &v[end][0], nx, MPI_FLOAT, rank + 1, 1,
                     MPI_COMM_WORLD, &status);
      }
      // p
      if (rank > 0) {
        MPI_Sendrecv(&p[begin][0], nx, MPI_FLOAT, rank - 1, 2,
                     &p[begin - 1][0], nx, MPI_FLOAT, rank - 1, 2,
                     MPI_COMM_WORLD, &status);
      }
      if (rank < size - 1) {
        MPI_Sendrecv(&p[end - 1][0], nx, MPI_FLOAT, rank + 1, 2,
                     &p[end][0], nx, MPI_FLOAT, rank + 1, 2,
                     MPI_COMM_WORLD, &status);
      }
    }
    for (int j=0; j<ny; j++) {
      u[j][0]    = 0.0; // u[:, 0] = 0
      u[j][nx-1] = 0.0; // u[:, -1] = 0
      v[j][0]    = 0.0; // v[:, 0] = 0
      v[j][nx-1] = 0.0; // v[:, -1] = 0
    }
    for (int i=0; i<nx; i++) {
      u[0][i]    = 0.0; // u[0, :] = 0
      u[ny-1][i] = 1.0; // u[-1, :] = 1
      v[0][i]    = 0.0; // v[0, :] = 0
      v[ny-1][i] = 0.0; // v[-1, :] = 0
    }
    if (n % 10 == 0) {
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          ufile << u[j][i] << " ";
      ufile << "\n";
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          vfile << v[j][i] << " ";
      vfile << "\n";
      for (int j=0; j<ny; j++)
        for (int i=0; i<nx; i++)
          pfile << p[j][i] << " ";
      pfile << "\n";
    }
  }
  ufile.close();
  vfile.close();
  pfile.close();
  MPI_Finalize();
}
