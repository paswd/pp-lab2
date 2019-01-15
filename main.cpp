/*
 ==================================================
| __________                 __      __________    |
| \______   \_____    ______/  \    /  \______ \   |
|  |     ___/\__  \  /  ___/\   \/\/   /|    |  \  |
|  |    |     / __ \_\___ \  \        / |    `   \ |
|  |____|    (____  /____  >  \__/\  / /_______  / |
|                \/     \/        \/          \/   |
 ==================================================
(c) Developed by Pavel Sushko (PasWD)
http://paswd.ru/
me@paswd.ru

Original repository:
https://github.com/paswd/pp-lab2

All rights reserved
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parallel programming in Moscow Aviation Institute
Laboratory work #2

Variant:
* 3D
* Jacoby method
* Alltoallv
* Allgather
* Cycle & barrier output
*/

#include "mpi.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

const size_t DEFAULT_STEP = 1;

const int TAGS_LAYER = 1;

class Puasson3dSolver {
private:
	//Input variables
	size_t SizeX;
	size_t SizeY;
	size_t SizeZ;

	size_t IterCnt;
	double Eps;
	double Coeff;

	//Math variables
	double StepSqX;
	double StepSqY;
	double StepSqZ;

	double ***Func;
	double ***NewFunc;
	double ***Ro;

	//Cluster variables
	size_t MachinesCnt;
	size_t MachineId;
	size_t LayersOnMachine;

	double *SendBuffer;
	double *GetBuffer;
	size_t ExistingMachinesCnt;
	int *BufferCounts;
	int *BufferDispls;

	double *EpsBuffer;
	int *EpsBufferCounts;
	int *EpsBufferDispls;

	size_t BufferSize;
	MPI_Status Status;

public:
	bool isUnclaimed() {
		return MachineId >= MachinesCnt;
	}

private:
	void initDefaultVars() {
		StepSqX = 1.;
		StepSqY = 1.;
		StepSqZ = 1.;

		SendBuffer = NULL;
		GetBuffer = NULL;
	}

	bool isFirstMachine() {
		return MachineId == 0;
	}

	bool isLastMachine() {
		return MachineId == (MachinesCnt - 1);
	}

	size_t getLayersOnMachine(size_t machineId) {
		size_t res = SizeZ / MachinesCnt;
		if (machineId < SizeZ % MachinesCnt) {
			res++;
		}
		return res;
	}

	size_t getOriginalZPos(size_t posInLayer) {
		size_t prevLayersCnt = 0;
		for (size_t i = 0; i < MachineId; i++) {
			prevLayersCnt += getLayersOnMachine(i);
		}
		size_t res = prevLayersCnt + posInLayer;
		if (!isFirstMachine()) {
			res--;
		}
		return res;
	}

	bool isBorder(size_t i, size_t j, size_t k) {
		size_t origK = getOriginalZPos(k);
		if (i == 0 || j == 0 || origK == 0) {
			return true;
		}
		if (i == SizeX - 1 || j == SizeY - 1 || origK == SizeZ - 1) {
			return true;
		}
		return false;
	}

	void setClusterParams(size_t machinesCnt, size_t machineId) {
		MachinesCnt = machinesCnt;
		MachineId = machineId;

		LayersOnMachine = getLayersOnMachine(MachineId);
		if (isFirstMachine() != isLastMachine()) {
			LayersOnMachine++;
		} else if (!isFirstMachine()) {
			LayersOnMachine += 2;
		}
	}

	void getDataFromFile(string filename) {
		ifstream in(filename.c_str());
		in >> SizeX >> SizeY >> SizeZ >>
			Eps >> Coeff;
		in.close();
		BufferSize = SizeX * SizeY;
	}

	void copyOneLayerToBuffer(double *buffer, size_t layerId) {
		for (size_t i = 0; i < SizeX; i++) {
			for (size_t j = 0; j < SizeY; j++) {
				buffer[(i * SizeX) + j] = Func[i][j][layerId];
			}
		}
	}

	void getOneLayerFromBuffer(double *buffer, size_t layerId) {
		for (size_t i = 0; i < SizeX; i++) {
			for (size_t j = 0; j < SizeY; j++) {
				Func[i][j][layerId] = buffer[(i * SizeX) + j];
			}
		}
	}

	double *getBufferElement(double *buffer, size_t pos) {
		size_t layerSize = SizeX * SizeY;
		return &(buffer[pos * layerSize]);
	}

	void copyLayersToBuffer(double *buffer) {
		if (!isFirstMachine()) {
			double *bufferElementPrev = getBufferElement(buffer, MachineId - 1);
			copyOneLayerToBuffer(bufferElementPrev, 1);
		}
		if (!isLastMachine()) {
			double *bufferElementNext = getBufferElement(buffer, MachineId + 1);
			copyOneLayerToBuffer(bufferElementNext, MachinesCnt - 2);
		}
	}

	void getLayersFromBuffer(double *buffer) {
		if (!isFirstMachine()) {
			double *bufferElementPrev = getBufferElement(buffer, MachineId - 1);
			getOneLayerFromBuffer(bufferElementPrev, 0);
		}
		if (!isLastMachine()) {
			double *bufferElementNext = getBufferElement(buffer, MachineId + 1);
			getOneLayerFromBuffer(bufferElementNext, MachinesCnt - 1);
		}
	}

	void arrInit() {
		Func = new double**[SizeX];
		NewFunc = new double**[SizeX];
		Ro = new double**[SizeX];

		for (size_t i = 0; i < SizeX; i++) {
			Func[i] = new double*[SizeY];
			NewFunc[i] = new double*[SizeY];
			Ro[i] = new double*[SizeY];

			for (size_t j = 0; j < SizeY; j++) {

				Func[i][j] = new double[LayersOnMachine];
				NewFunc[i][j] = new double[LayersOnMachine];
				Ro[i][j] = new double[LayersOnMachine];

				for (size_t k = 0; k < LayersOnMachine; k++) {
					if (isBorder(i, j, k)) {
						Func[i][j][k] = 1.;
						NewFunc[i][j][k] = 1.;
					} else {
						Func[i][j][k] = 0.;
						NewFunc[i][j][k] = 0.;
					}
					Ro[i][j][k] = 0.;
				}
			}
		}

		SendBuffer = new double[BufferSize];
		GetBuffer = new double[BufferSize];

		BufferCounts = new int[ExistingMachinesCnt];
		BufferDispls = new int[ExistingMachinesCnt];

		size_t layerSize = SizeX * SizeY;
		size_t currentDispl = 0;

		for (size_t i = 0; i < ExistingMachinesCnt; i++) {
			BufferCounts[i] = (int) layerSize;
			BufferDispls[i] = (int) currentDispl;
			currentDispl += layerSize;
		}

		EpsBuffer = new double[ExistingMachinesCnt];
		EpsBufferCounts = new int[ExistingMachinesCnt];
		EpsBufferDispls = new int[ExistingMachinesCnt];
		currentDispl = 0;

		for (size_t i = 0; i < ExistingMachinesCnt; i++) {
			EpsBufferCounts[i] = 1;
			EpsBufferDispls[i] = currentDispl;
			currentDispl++;
		}
	}

	void arrClear() {
		for (size_t i = 0; i < SizeX; i++) {
			for (size_t j = 0; j < SizeY; j++) {
				delete [] Func[i][j];
				delete [] NewFunc[i][j];
				delete [] Ro[i][j];
			}

			delete [] Func[i];
			delete [] NewFunc[i];
			delete [] Ro[i];
		}

		delete [] Func;
		delete [] NewFunc;
		delete [] Ro;

		delete [] SendBuffer;
		delete [] GetBuffer;

		delete [] BufferCounts;
		delete [] BufferDispls;
	}

	double getElementNewValue(size_t i, size_t j, size_t k) {
		return (
				(Func[i + 1][j][k] + Func[i - 1][j][k]) / (double) StepSqX +
				(Func[i][j + 1][k] + Func[i][j - 1][k]) / (double) StepSqY +
				(Func[i][j][k + 1] + Func[i][j][k - 1]) / (double) StepSqZ -
				Ro[i][j][k]
			) / (
				2. / (double) StepSqX +
				2. / (double) StepSqY +
				2. / (double) StepSqZ +
				Coeff
			);
	}

	void machineExchange() {
		copyLayersToBuffer(SendBuffer);
		MPI_Alltoallv(SendBuffer, BufferCounts, BufferDispls, MPI_DOUBLE,
			GetBuffer, BufferCounts, BufferDispls, MPI_DOUBLE, MPI_COMM_WORLD);
		getLayersFromBuffer(GetBuffer);
	}

	double getCurrEps(double ***f1, double ***f2, size_t untillX, size_t untillY, size_t untillZ) {
		double resEps = Eps;
		if (isUnclaimed()) {
			return 0.;
		}
		for (size_t i = 0; i < untillX; i++) {
			for (size_t j = 0; j < untillY; j++) {
				for (size_t k = 0; k < untillZ; k++) {
					double currEps = abs(f1[i][j][k] - f2[i][j][k]);
					if (i == 0 && j == 0 && k == 0) {
						resEps = currEps;
					} else {
						resEps = max(resEps, currEps);
					}
				}
			}
		}
		return resEps;
	}
	double getAllMachinesEps(double currEps) {
		MPI_Allgatherv(&currEps, 1, MPI_DOUBLE, EpsBuffer, EpsBufferCounts, EpsBufferDispls,
			MPI_DOUBLE, MPI_COMM_WORLD);
		double maxEps = currEps;

		for (size_t i = 0; i < ExistingMachinesCnt; i++) {
			if (EpsBuffer[i] > maxEps) {
				maxEps = EpsBuffer[i];
			}
		}

		return maxEps;
	}

public:
	Puasson3dSolver(string filename, size_t machinesCnt, size_t machineId) {
		initDefaultVars();
		getDataFromFile(filename);
		ExistingMachinesCnt = machinesCnt;
		setClusterParams(min(machinesCnt, SizeZ), machineId);
		BufferSize *= ExistingMachinesCnt;
		arrInit();
	}
	~Puasson3dSolver() {
		arrClear();
	}

	void solve() {
		size_t untillX = SizeX - 1;
		size_t untillY = SizeY - 1;
		size_t untillZ = LayersOnMachine - 1;
		double currEps = Eps * 2;
		IterCnt = 0;

		while (currEps > Eps) {
			machineExchange();

			for (size_t i = 1; i < untillX; i++) {
				for (size_t j = 1; j < untillY; j++) {
					for (size_t k = 1; k < untillZ; k++) {
						NewFunc[i][j][k] = getElementNewValue(i, j, k);
					}
				}
			}
			currEps = getAllMachinesEps(getCurrEps(Func, NewFunc, untillX, untillY, untillZ));

			double ***tmp = Func;
			Func = NewFunc;
			NewFunc = tmp;
			IterCnt++;
		}
	}

	void writeResult(std::ostream& out) {
		for (size_t i = 0; i < MachinesCnt; i++) {
			if (i == MachineId) {
				if (MachineId == 0) {
					out << "Solution for " << SizeX << "x" << SizeY << "x" << SizeZ << endl;
					out << "Alpha = " << Coeff << endl;
					out << "Eps: " << Eps << endl;
					out << "Iterations count: " << IterCnt << endl << endl;
				}
				for (size_t k = 0; k < LayersOnMachine; k++) {
					if (k == 0 && !isFirstMachine()) {
						continue;
					}
					if (k == LayersOnMachine - 1 && !isLastMachine()) {
						continue;
					}
					out << "k = " << getOriginalZPos(k) << endl;

					for (size_t j = 0; j < SizeY; j++) {
						for (size_t i = 0; i < SizeX; i++) {
							if (i > 0) {
								out << " ";
							}
							out << round(Func[i][j][k] * 100.) / 100.;
						}
						out << endl;
					}
					out << endl;
				}
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}
	}
};


int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	int machineId, machinesCnt;
	MPI_Comm_size(MPI_COMM_WORLD, &machinesCnt);
	MPI_Comm_rank(MPI_COMM_WORLD, &machineId);

	Puasson3dSolver solver("input.txt", machinesCnt, machineId);
	if (!solver.isUnclaimed()) {
		solver.solve();
		solver.writeResult(std::cout);
	}
	MPI_Finalize();

	return 0;
}
