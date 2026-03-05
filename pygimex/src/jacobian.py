def createJacobian(self, model):
        """Create Jacobian matrix for model and store it in self.jacobian()."""
        if self.subPotentials is None:
            self.response(model)

        J = self.jacobian()
        J.resize(self.data.size(), self.parameterCount)

        cells = self.mesh().findCellByMarker(0, -1)
        Si = pg.matrix.ElementMatrix()
        St = pg.matrix.ElementMatrix()

        u = self.subPotentials

        pg.tic()
        if self.verbose:
            print("Calculate sensitivity matrix for model: ",
                  min(model), max(model))

        Jt = pg.Matrix(self.data.size(), self.parameterCount)

        for kIdx, w in enumerate(self.w):
            k = self.k[kIdx]
            w = self.w[kIdx]

            Jt *= 0.
            A = pg.matrix.ElementMatrixMap()

            for i, c in enumerate(cells):
                modelIdx = c.marker()

                # 2.5D
                Si.u2(c)
                Si *= k * k
                Si += St.ux2uy2uz2(c)

                # 3D
                # Si.ux2uy2uz2(c); w = w* 2

                A.add(modelIdx, Si)

            for dataIdx in range(self.data.size()):

                a = int(self.data['a'][dataIdx])
                b = int(self.data['b'][dataIdx])
                m = int(self.data['m'][dataIdx])
                n = int(self.data['n'][dataIdx])
                Jt[dataIdx] = A.mult(u[kIdx][a] - u[kIdx][b],
                                     u[kIdx][m] - u[kIdx][n])

            J += w * Jt

        m2 = model*model
        k = self.data['k']

        for i in range(J.rows()):
            J[i] /= (m2 / k[i])

        if self.verbose:
            sumsens = np.zeros(J.rows())
            for i in range(J.rows()):
                sumsens[i] = pg.sum(J[i])
            print("sens sum: median = ", pg.math.median(sumsens),
                  " min = ", pg.min(sumsens),
                  " max = ", pg.max(sumsens))