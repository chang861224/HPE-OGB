#ifndef BPR_H
#define BPR_H

#include "../smore/src/proNet.h"

/*****
 * BPR
 * **************************************************************/

class BPR {

    public:
        
        BPR();
        ~BPR();
        
        proNet pnet;

        // parameters
        int dim;                // representation dimensions
        vector< vector<double> > w_vertex;

        // data function
        void LoadEdgeList(string, bool);
        void SaveWeights(string);
        
        // model function
        void Init(int, string, bool);
        void Train(int, int, double, double, int);

};


#endif
