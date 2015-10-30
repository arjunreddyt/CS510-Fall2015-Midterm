from attractor import Attractor
from nose import with_setup
import numpy as np

class TestCases:
    def setup(self):
        self.myObj = Attractor()
        

    def test_euler(self):
        assert self.myObj.euler(np.array([0.1,0.,0.]))[0]==-1.
        assert int(self.myObj.euler(np.array([0.1,0.,0.]))[1])==int(2.8)
        assert self.myObj.euler(np.array([0.1,0.,0.]))[2]==0.
      
    
    def test_rk2(self):
        obj = Attractor()
        obj.evolve([10,10,10],2)
        assert attr.solution['x'].count() > 0
        assert attr.solution['y'].count() > 0
        assert attr.solution['z'].count() > 0

        

