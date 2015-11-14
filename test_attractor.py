from attractor import Attractor
from nose import with_setup
import numpy as np
import os

class TestCases:
    
    def setup(self):
        self.s=s
        self.p=p
        self.b=b
        start=0
        self.start=start
        self.end=end
        self.points=points
        self.a=Attractor(s,p,b,start,end,points)


    def test_euler(self):
        "To Test Euler Method"
        assert self.myObj.euler(np.array([0.1,0.,0.]))[0]==-1.
        assert int(self.myObj.euler(np.array([0.1,0.,0.]))[1])==int(2.8)
        assert self.myObj.euler(np.array([0.1,0.,0.]))[2]==0.


    def test_rk2(self):
        "To Order 2nd Order Runga-Kutta"
        obj = Attractor()
        obj.evolve([10,10,10],2)
        assert attr.solution['x'].count() > 0
        assert attr.solution['y'].count() > 0
        assert attr.solution['z'].count() > 0

    def test_solve(self):
        "To Test saving to .csv function"
            self.a.save()
            data=open('save_solution.csv','r')
            d=data.read()

    def test(self):
        "To test attractor result"
            obj = Attractor()
            expected_result =  [9.6 , 5.6 , 19.97333333]
            res = obj.euler([10,5,20])
            assert (expected_result == res).all
            print "Passed"


