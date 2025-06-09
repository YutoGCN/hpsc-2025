#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>
#include <mpi.h>

using namespace std;
typedef vector<vector<float>> matrix;

int main(int argc, char** argv) {
  int nx = 41;
  int ny = 41;
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
  int local_ny = end - begin;
  int local_ny_g = local_ny + 2;

  matrix u(local_ny_g, vector<float>(nx, 0.0));
  matrix v(local_ny_g, vector<float>(nx, 0.0));
  matrix p(local_ny_g, vector<float>(nx, 0.0));
  matrix b(local_ny_g, vector<float>(nx, 0.0));
  matrix un(local_ny_g, vector<float>(nx, 0.0));
  matrix vn(local_ny_g, vector<float>(nx, 0.0));
  matrix pn(local_ny_g, vector<float>(nx, 0.0));

  ofstream ufile, vfile, pfile;
  if (rank == 0) {
    ufile.open("u.dat");
    vfile.open("v.dat");
    pfile.open("p.dat");
  }

  for (int n = 0; n < nt; n++) {
    // bの計算
    for (int j = 1; j <= local_ny; j++) {
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

    for (int it = 0; it < nit; it++) {
      for (int j = 0; j < local_ny_g; j++)
        for (int i = 0; i < nx; i++)
          pn[j][i] = p[j][i];

      for (int j = 1; j <= local_ny; j++) {
        for (int i = 1; i < nx - 1; i++) {
          p[j][i] = (dy * dy * (pn[j][i + 1] + pn[j][i - 1]) +
                     dx * dx * (pn[j + 1][i] + pn[j - 1][i]) -
                     b[j][i] * dx * dx * dy * dy)
                    / (2.0 * (dx * dx + dy * dy));
        }
      }

      for (int j = 1; j <= local_ny; j++) {
        p[j][nx - 1] = p[j][nx - 2];
        p[j][0] = p[j][1];
      }
      if (begin == 0) {
        for (int i = 0; i < nx; i++) {
          p[1][i] = p[2][i]; 
        }
      }
      if (end == ny) {
        for (int i = 0; i < nx; i++) {
          p[local_ny][i] = 0.0; 
        }
      }

      MPI_Status status;
      if (rank > 0) {
        MPI_Sendrecv(&p[1][0], nx, MPI_FLOAT, rank - 1, 0,
                     &p[0][0], nx, MPI_FLOAT, rank - 1, 0,
                     MPI_COMM_WORLD, &status);
      }
      if (rank < size - 1) {
        MPI_Sendrecv(&p[local_ny][0], nx, MPI_FLOAT, rank + 1, 0,
                     &p[local_ny + 1][0], nx, MPI_FLOAT, rank + 1, 0,
                     MPI_COMM_WORLD, &status);
      }
    }

    // un, vn = u, v
    for (int j = 0; j < local_ny_g; j++) {
      for (int i = 0; i < nx; i++) {
        un[j][i] = u[j][i];
        vn[j][i] = v[j][i];
      }
    }

    for (int j = 1; j <= local_ny; j++) {
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

    MPI_Status status;
    if (rank > 0) {
      MPI_Sendrecv(&u[1][0], nx, MPI_FLOAT, rank - 1, 1,
                   &u[0][0], nx, MPI_FLOAT, rank - 1, 1,
                   MPI_COMM_WORLD, &status);
      MPI_Sendrecv(&v[1][0], nx, MPI_FLOAT, rank - 1, 2,
                   &v[0][0], nx, MPI_FLOAT, rank - 1, 2,
                   MPI_COMM_WORLD, &status);
    }
    if (rank < size - 1) {
      MPI_Sendrecv(&u[local_ny][0], nx, MPI_FLOAT, rank + 1, 1,
                   &u[local_ny + 1][0], nx, MPI_FLOAT, rank + 1, 1,
                   MPI_COMM_WORLD, &status);
      MPI_Sendrecv(&v[local_ny][0], nx, MPI_FLOAT, rank + 1, 2,
                   &v[local_ny + 1][0], nx, MPI_FLOAT, rank + 1, 2,
                   MPI_COMM_WORLD, &status);
    }

    for (int j = 1; j <= local_ny; j++) {
      u[j][0] = 0.0;
      u[j][nx - 1] = 0.0;
      v[j][0] = 0.0;
      v[j][nx - 1] = 0.0;
    }
    if (begin == 0) {
      for (int i = 0; i < nx; i++) {
        u[1][i] = 0.0; 
        v[1][i] = 0.0;
      }
    }
    if (end == ny) {
      for (int i = 0; i < nx; i++) {
        u[local_ny][i] = 1.0; 
        v[local_ny][i] = 0.0;
      }
    }

    if (n % 10 == 0) {
      std::vector<float> u_local(local_ny * nx);
      std::vector<float> v_local(local_ny * nx);
      std::vector<float> p_local(local_ny * nx);
      for (int j = 1; j <= local_ny; j++)
        for (int i = 0; i < nx; i++) {
          u_local[(j - 1) * nx + i] = u[j][i];
          v_local[(j - 1) * nx + i] = v[j][i];
          p_local[(j - 1) * nx + i] = p[j][i];
        }

      std::vector<float> u_all, v_all, p_all;
      if (rank == 0) {
        u_all.resize(ny * nx);
        v_all.resize(ny * nx);
        p_all.resize(ny * nx);
      }
      std::vector<int> recvcounts, displs;
      if (rank == 0) {
        recvcounts.resize(size);
        displs.resize(size);
        for (int r = 0; r < size; r++) {
          int b = r * ny / size;
          int e = (r + 1) * ny / size;
          recvcounts[r] = (e - b) * nx;
          displs[r] = b * nx;
        }
      }
      MPI_Gatherv(u_local.data(), local_ny * nx, MPI_FLOAT,
                  rank == 0 ? u_all.data() : nullptr,
                  rank == 0 ? recvcounts.data() : nullptr,
                  rank == 0 ? displs.data() : nullptr,
                  MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Gatherv(v_local.data(), local_ny * nx, MPI_FLOAT,
                  rank == 0 ? v_all.data() : nullptr,
                  rank == 0 ? recvcounts.data() : nullptr,
                  rank == 0 ? displs.data() : nullptr,
                  MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Gatherv(p_local.data(), local_ny * nx, MPI_FLOAT,
                  rank == 0 ? p_all.data() : nullptr,
                  rank == 0 ? recvcounts.data() : nullptr,
                  rank == 0 ? displs.data() : nullptr,
                  MPI_FLOAT, 0, MPI_COMM_WORLD);

      if (rank == 0) {
        for (int j = 0; j < ny; j++)
          for (int i = 0; i < nx; i++)
            ufile << u_all[j * nx + i] << " ";
        ufile << "\n";
        for (int j = 0; j < ny; j++)
          for (int i = 0; i < nx; i++)
            vfile << v_all[j * nx + i] << " ";
        vfile << "\n";
        for (int j = 0; j < ny; j++)
          for (int i = 0; i < nx; i++)
            pfile << p_all[j * nx + i] << " ";
        pfile << "\n";
      }
    }
  }
  if (rank == 0) {
    ufile.close();
    vfile.close();
    pfile.close();
  }
  MPI_Finalize();
}
