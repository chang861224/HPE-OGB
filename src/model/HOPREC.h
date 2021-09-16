#ifndef HOPREC_H
#define HOPREC_H

#include "../../smore/src/model/LINE.h"

/*****
 * HOPREC
 * **************************************************************/

class HOPREC {

    public:
        
        HOPREC();
        ~HOPREC();
        
        proNet pnet;

        // parameters
        int dim;                // representation dimensions
        vector< vector<double> > w_vertex;
        vector< vector<double> > w_context;

        // data function
        void LoadEdgeList(string, bool);
        void LoadFieldMeta(string);
        void SaveWeights(string);
        
        // model function
        void Init(int, string);
        void Train(int, int, double, int);

};


#endif
