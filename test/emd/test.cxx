/************************************************************************/
/*                                                                      */
/*               Copyright 2011-2013 by Ullrich Koethe                  */
/*                                                                      */
/*    This file is part of the VIGRA computer vision library.           */
/*    The VIGRA Website is                                              */
/*        http://hci.iwr.uni-heidelberg.de/vigra/                       */
/*    Please direct questions, bug reports, and contributions to        */
/*        ullrich.koethe@iwr.uni-heidelberg.de    or                    */
/*        vigra@informatik.uni-hamburg.de                               */
/*                                                                      */
/*    Permission is hereby granted, free of charge, to any person       */
/*    obtaining a copy of this software and associated documentation    */
/*    files (the "Software"), to deal in the Software without           */
/*    restriction, including without limitation the rights to use,      */
/*    copy, modify, merge, publish, distribute, sublicense, and/or      */
/*    sell copies of the Software, and to permit persons to whom the    */
/*    Software is furnished to do so, subject to the following          */
/*    conditions:                                                       */
/*                                                                      */
/*    The above copyright notice and this permission notice shall be    */
/*    included in all copies or substantial portions of the             */
/*    Software.                                                         */
/*                                                                      */
/*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND    */
/*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES   */
/*    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND          */
/*    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT       */
/*    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,      */
/*    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING      */
/*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR     */
/*    OTHER DEALINGS IN THE SOFTWARE.                                   */
/*                                                                      */
/************************************************************************/

#include <unittest.hxx>
#include <vigra/emd.hxx>
#include <vigra/random.hxx>
#include <vigra/error.hxx>

#include "emd.hxx"

using namespace vigra;

class EarthMoverDistanceTest
{
    RandomMT19937 random;

    // 3d point type used for example1 from original source code
    // see also testEMD_RTG_example1()
    struct point3d {
        int X,Y,Z;
    };

public:
    const double errorTolerance;

    EarthMoverDistanceTest() : errorTolerance(1e-6)
    {
        // tests should be deterministic
        random.seed(0);
    }

    /*******************************************************************/
    // utility functions

    // Check that the flow given by Flow between signatures Signature1 and
    // Signature2 satisfies some basic properties.
    //
    // Let Signature1 and Signature2 have labels l_i and l_j with
    // corresponding weights w_i and w_j respectively.
    //
    // Consider the flow as a list of arrows each with origin i, target
    // j and weight f_ij. Then
    //
    // 1) f_ij >= 0 for all i, j
    //      the flow is only in one direction
    // 2) \sum_i f_ij <= w_i
    //      sum of arrows originating in the i-th coordinate is <= w_i
    // 3) \sum_j f_ij <= w_j
    //      sum of arrows ending in the j-th coordinate is <= w_j
    // 4) \sum_{i,j} f_ij = min(\sum_i w_i, \sum_j w_j)
    //      sum of the total flow does not exceed total available earth or
    //      space
    //
    // See page 8 of the following text for further explanation:
    //
    // http://www.cs.cmu.edu/~efros/courses/AP06/Papers/rubner-jcviu-00.pdf
    template<typename feature_t>
    void checkFlowProperties(const Signature<feature_t> &signature1,
            const Signature<feature_t> &signature2,
            flow_t *Flow, int FlowSize)
    {
        double source_total = 0;
        double dest_total = 0;
        double total_flow = 0;
        // Following loop
        //  - checks for positive flow,
        //  - computes total flow,
        //  - makes sure the origin and destination labels are valid
        //  - verifies that there are no duplicate arrows in the flow
        for (int i=0; i < FlowSize; ++i)
        {
            shouldMsg(Flow[i].amount >= 0, "flow in the wrong direction");
            total_flow += Flow[i].amount;

            shouldMsg(Flow[i].from >=0 && Flow[i].from < signature1.size(),
                    "Flow origin label not found in Signature");
            shouldMsg(Flow[i].to >=0 && Flow[i].to < signature2.size(),
                    "Flow destination label not found in Signature");
            for (int j=i+1; j < FlowSize; ++j)
            {
                if(Flow[i].from == Flow[j].from && Flow[i].to == Flow[j].to)
                    shouldMsg(false,
                            "flow contains duplicate arrows");
            }
        }

        // compute total initial and target weights
        for (int i=0; i < signature1.size(); ++i)
        {
            source_total += signature1.weights_[i];
        }
        for (int i=0; i < signature2.size(); ++i)
        {
            dest_total += signature2.weights_[i];
        }
        shouldEqualToleranceMessage(std::min(source_total, dest_total),
                total_flow, errorTolerance,
                "total flow more than minimum of total initial or target");

        // check if source bins are used within their limits
        for (int i=0; i < signature1.size(); ++i)
        {
            double tot = 0;
            for (int j=0; j < FlowSize; ++j)
            {
                if(Flow[j].from == i)
                    tot += Flow[j].amount;
            }
            // FIXME: floats are notoriously error prone
            // the scaling factor below should at least be computed based
            // on precision and size of source/target signatures
            shouldMsg(tot <= signature1.weights_[i] + errorTolerance*1e4,
                    "flow uses more earth than available in bin");
        }

        // check if target bins are not filled over their capacity
        for (int i=0; i < signature2.size(); ++i)
        {
            double tot = 0;
            for (int j=0; j < FlowSize; ++j)
            {
                if(Flow[j].to == i)
                    tot += Flow[j].amount;
            }
            // FIXME: floats are notoriously error prone
            // the scaling factor below should at least be computed based
            // on precision and size of source/target signatures
            shouldMsg(tot <= signature2.weights_[i] + errorTolerance*1e4,
                    "overflow in target bin");
        }
    }

    void printFlow(flow_t *flow, int flowSize)
    {
        std::cout<<"flow: "<<std::endl;
        std::cout<<"from\tto\tamount"<<std::endl;
        for (int i=0; i < flowSize; i++)
            if (flow[i].amount > 0)
                std::cout<<flow[i].from<<" "<<flow[i].to<<" "<<flow[i].amount<<std::endl;
    }

    /*******************************************************************/
    // distance functions
    //
    // temporary workaround distance functions which work with the
    // definition in the original emd sources.
    // Should be replaced by a functor.

    // L1 norm in one dimension
    static double dist_l1(const int &F1, const int &F2)
    {
        return std::abs(F1 - F2);
    }

    // L2 distance on 3-dimensional space defined by the original example1
    static double dist_example1(const point3d& F1, const point3d& F2)
    {
        int dX = F1.X - F2.X, dY = F1.Y - F2.Y, dZ = F1.Z - F2.Z;
        return sqrt(dX*dX + dY*dY + dZ*dZ);
    }

    // Matrix based distance as defined by the original example2
    static double dist_example2(const int &F1, const int &F2)
    {
        // Cost matrix as defined in example2 of original sources
        // http://robotics.stanford.edu/~rubner/emd/default.htm
        static double _COST[5][3] = {
            {3, 5, 2},
            {0, 2, 5},
            {1, 1, 3},
            {8, 4, 3},
            {7, 6, 5}
            };
        return _COST[F1][F2];
    }

    // end distance functions
    /*******************************************************************/

    // example 1 on
    // http://robotics.stanford.edu/~rubner/emd/default.htm
    void testEMD_RTG_example1()
    {
        point3d f1[4] = { {100,40,22}, {211,20,2}, {32,190,150}, {2,100,100} },
                f2[3] = { {0,0,0}, {50,100,80}, {255,255,255} };
        double w1[5] = { 0.4, 0.3, 0.2, 0.1 },
              w2[3] = { 0.5, 0.3, 0.2 };
        Signature<point3d> s1(f1, w1, 4), s2(f2, w2, 3);
        double e;

        e = emd<point3d>(s1, s2, dist_example1, 0, 0);
        shouldEqualToleranceMessage(e, 160.542770, errorTolerance,
                "Earth Moving Distance not within tolerance");
    }

    // example 2 on
    // http://robotics.stanford.edu/~rubner/emd/default.htm
    void testEMD_RTG_example2()
    {
        int f1[5] = { 0, 1, 2, 3, 4 }, f2[3] = { 0, 1, 2 };
        double w1[5] = { 0.4, 0.2, 0.2, 0.1, 0.1 },
               w2[3] = { 0.6, 0.2, 0.1 };
        Signature<int> s1(f1, w1, 5), s2(f2, w2, 3);

        double   e;
        flow_t      flow[7];
        int         flowSize;

        /* original source returns
         *
         * emd=1.888889
         *
         * flow:
         * from to  amount
         * 1    0   0.200000
         * 0    0   0.300000
         * 2    0   0.100000
         * 3    1   0.100000
         * 2    1   0.100000
         * 0    2   0.100000
         */
        e = emd<int>(s1, s2, dist_example2, flow, &flowSize);
        shouldEqualMessage(flowSize, 6, "flow should have 6 items");
        shouldEqualToleranceMessage(e, 1.88889, errorTolerance,
                "Earth Moving Distance not within tolerance");
        checkFlowProperties(s1, s2, flow, flowSize);
    }

    void testEMDEpsilon()
    {
        int sourceFeatures[1] = {482}, targetFeatures[94];
        for (int i=0; i < 94; ++i)
            targetFeatures[i] = i;
        double sourceWeights[1] = {883.796822951128},
               targetWeights[94] = {2.51323772263489, 10.1437428111339,
                   12.2601341055487, 4.92835956990509, 6.0069893043799,
                   12.1400001931411, 14.8478551315396, 10.3666762361407,
                   2.40344903151515, 13.4098880314272, 9.49754202045642,
                   6.62167720122611, 18.3624799356215, 10.2093360351609,
                   13.5676855017886, 15.1861955670949, 2.37642144026657,
                   15.1583532879013, 12.3073085410011, 13.3548176060718,
                   12.553736067709, 10.1330857345309, 15.440411475033,
                   12.5885970070129, 5.54827417123641, 1.62448264269099,
                   2.74523794672398, 8.81266169758666, 15.8318190070725,
                   16.0829133840968, 6.82208084939348, 11.3564722684034,
                   8.21353797301209, 17.6075360794372, 17.9467924816885,
                   4.9021987817979, 8.57284119451089, 1.80064552828248,
                   1.52601175622286, 11.716381634346, 12.3598628441529,
                   1.70012775663702, 13.4022904664341, 14.6243217570538,
                   8.73716323408799, 4.80055461490972, 13.1449366107076,
                   10.0735510120077, 7.40597827325194, 15.1710807931603,
                   16.8402430026618, 14.7346395364838, 14.4085701813759,
                   9.27371237602036, 5.22251962308536, 18.2463196600088,
                   8.82826515107264, 9.30471594186857, 13.5036488160148,
                   5.21282577862715, 9.98796859540222, 5.65484192032304,
                   9.72180414135422, 13.9045884486405, 15.7537370560249,
                   8.17300203666348, 12.2042053271059, 7.8304828034458,
                   17.5848735266079, 0.383057233103421, 5.07810212752952,
                   5.70807341628602, 1.86479732373033, 16.0873930402498,
                   3.47750065212404, 15.0238805089369, 3.54938724213822,
                   13.9582167096831, 10.3623763166845, 0.43245327089761,
                   9.57186199092982, 0.504662887189582, 11.1069465191737,
                   5.82375377149663, 3.75934330410765, 8.87129988287845,
                   4.47410674004744, 17.0102896033561, 1.06496512650127,
                   7.19178601333074, 0.473460690654718, 5.39482381612472,
                   6.64620145201165, 10.6793850720285};
        Signature<int> sourceSignature(sourceFeatures, sourceWeights, 1),
                    targetSignature(targetFeatures, targetWeights, 94);

        flow_t flow[94];
        int flowSize;

        EMDOptions options = EMDOptions();
        options.setEpsilon(0);

        bool raised = false;
        try
        {
        emd(sourceSignature, targetSignature, dist_l1, flow, &flowSize,
                options);
        }
        catch(std::runtime_error &c)
        {
            raised = true;
            std::string expected("\nemd: Unexpected error in findBasicVariables!\nThis typically happens when epsilon defined in\nEMDOptions not right for the scale of the problem.");
            std::string message(c.what());
            shouldMsg(0 == message.compare(0, expected.length(), expected),
                    "No error raised for empty source signature");
        }
        if (!raised)
            shouldMsg(false, "epsilon=0 should not produce a valid flow for this example.");
    }
    // Test if emd() behaves properly for empty and too large source or
    // target signatures.
    void testEMDEmptyInOut()
    {
        int emptyFeatures[0] = {},
            zeroFeatures[5] = {0, 1, 2, 3, 4},
            nonemptyFeatures[3] = { 0, 1, 2 };
        double emptyWeights[0] = {},
               zeroWeights[5] = {0, 0, 0, 0, 0},
               nonemptyWeights[3] = { 1, 5, 8 };
        Signature<int> emptySignature,
            zeroSignature(zeroFeatures, zeroWeights, 5),
            nonemptySignature(nonemptyFeatures, nonemptyWeights, 3);

        flow_t      flow[7]; // flow size is bounded by sig1->n + sig2->n - 1
        int         flowSize;

        // Empty signatures
        try
        {
        emd<int>(emptySignature, emptySignature, dist_l1, flow, &flowSize);
        }
        catch(vigra::ContractViolation &c)
        {
            std::string expected("\nPrecondition violation!\nSource signature cannot be empty!");
            std::string message(c.what());
            shouldMsg(0 == message.compare(0, expected.length(), expected),
                    "No error raised for empty source signature");
        }

        try
        {
        emd(nonemptySignature, emptySignature, dist_l1, flow, &flowSize);
        }
        catch(vigra::ContractViolation &c)
        {
            std::string expected("\nPrecondition violation!\nTarget signature cannot be empty!");
            std::string message(c.what());
            shouldMsg(0 == message.compare(0, expected.length(), expected),
                    "No error raised for empty target signature");
        }

        try
        {
        emd(emptySignature, nonemptySignature, dist_l1, flow, &flowSize);
        }
        catch(vigra::ContractViolation &c)
        {
            std::string expected("\nPrecondition violation!\nSource signature cannot be empty!");
            std::string message(c.what());
            shouldMsg(0 == message.compare(0, expected.length(), expected),
                    "No error raised for empty source signature");
        }

        // Too large signatures
        EMDOptions options = EMDOptions();
        Signature<int> bigSignature(options.maxSigSize+1);
        try
        {
        emd(bigSignature, nonemptySignature, dist_l1, NULL, NULL);
        }
        catch(vigra::ContractViolation &c)
        {
            std::string expected("\nPrecondition violation!\nemd: Signature size is limited to ");
            std::string message(c.what());
            shouldMsg(0 == message.compare(0, expected.length(), expected),
                    "No error raised for too big source signature");
        }
        try
        {
        emd(nonemptySignature, bigSignature, dist_l1, NULL, NULL);
        }
        catch(vigra::ContractViolation &c)
        {
            std::string expected("\nPrecondition violation!\nemd: Signature size is limited to ");
            std::string message(c.what());
            shouldMsg(0 == message.compare(0, expected.length(), expected),
                    "No error raised for too big target signature.");
        }

        // Zero filled signatures
        try
        {
        emd(zeroSignature, zeroSignature, dist_l1, flow, &flowSize);
        }
        catch(vigra::ContractViolation &c)
        {
            std::string expected("\nPrecondition violation!\nTotal weight of source signature cannot be 0!");
            std::string message(c.what());
            shouldMsg(0 == message.compare(0, expected.length(), expected),
                    "No error raised for zero filled source signature.");
        }
        try
        {
        emd(zeroSignature, nonemptySignature, dist_l1, flow, &flowSize);
        }
        catch(vigra::ContractViolation &c)
        {
            std::string expected("\nPrecondition violation!\nTotal weight of source signature cannot be 0!");
            std::string message(c.what());
            shouldMsg(0 == message.compare(0, expected.length(), expected),
                    "No error raised for zero filled source and target  signatures.");
        }
        try
        {
        emd(nonemptySignature, zeroSignature, dist_l1, flow, &flowSize);
        }
        catch(vigra::ContractViolation &c)
        {
            std::string expected("\nPrecondition violation!\nTotal weight of target signature cannot be 0!");
            std::string message(c.what());
            shouldMsg(0 == message.compare(0, expected.length(), expected),
                    "No error raised for zero filled target signature.");
        }
    }

    // Test emd() between identical randomly generated input - output
    // signatures.
    void testEMDRandomToSelf()
    {
        // max bin sizes
        int binBounds[] = {10, 15, 20, 50, 100};
        // how many random signatures should be generated for each size
        const int nTriesPerSize = 100;
        // total weight of the signature will be numberOfBins * weightFactor
        const double weightFactor = 100;

        double e;
        flow_t* flow;
        int flowSize;
        Signature<int> sig;

        // Mighty compiler gods, make C++11 available, save us from our sins
        int lenBinBounds = sizeof(binBounds)/sizeof(int);
        //std::vector<int> v_binBounds(binBounds,
        //        binBounds + sizeof(binBounds)/sizeof(int));
        //for(std::vector<int>::const_iterator itr = v_binBounds.begin();
        //        itr != v_binBounds.end(); ++itr)
        for (int i=0; i < lenBinBounds; ++i)
        {
            int cBound = binBounds[i];
            for (int j=0; j < nTriesPerSize; ++j)
            {
                // generateRandomSignature allocates memory for sig
                sig.randomize(cBound * weightFactor, cBound, random=random);
                // flow size is bounded by n1 + n2 - 1 where n1 and n2 is the
                // number of bins in the source and target signatures
                // respectively
                flow = new flow_t[cBound*2 - 1];
                e = emd(sig, sig, dist_l1, flow, &flowSize);
                shouldMsg(e == 0, "EMD to self should be 0");
                shouldMsg(flowSize == sig.size(),
                        "Flow to self should have one arrow for each bin in the signature.");
                checkFlowProperties(sig, sig, flow, flowSize);

                // free stuff
                delete [] flow;
            }
        }
    }

    // Test if emd() is symmetric  for randomly generated input pairs.
    //
    // If s1 and s2 are signatures with equal total weight, then
    // emd(s1, s2) == emd(s2, s1) since emd() is a metric.
    void testEMDRandomSymmetric()
    {
        // max bin sizes
        int binBounds[] = {10, 15, 20, 50, 100};
        // how many random signatures should be generated for each size
        const int nTriesPerSize = 10;
        // total weight of the signature will be numberOfBins * weightFactor
        const double weightFactor = 1;

        double e1, e2;
        flow_t *oFlow, *rFlow;
        int oFlowSize, rFlowSize;
        Signature<int> sSig, dSig;

        int lenBinBounds = sizeof(binBounds)/sizeof(int);
        for (int i=0; i < lenBinBounds; ++i)
        {
            int sBound = binBounds[i];
            for (int j=0; j < lenBinBounds; ++j)
            {
                int dBound = binBounds[j];
                // flow size is bounded by sBound + dBound - 1
                oFlow = new flow_t[sBound + dBound  - 1];
                rFlow = new flow_t[sBound + dBound  - 1];
                for (int k=0; k < nTriesPerSize; ++k)
                {
                    // generateRandomSignature allocates memory for sig
                    sSig.randomize(sBound*dBound * weightFactor, sBound,
                            random);
                    dSig.randomize(sBound*dBound * weightFactor, dBound,
                            random);
                    e1 = emd(sSig, dSig, dist_l1, oFlow, &oFlowSize);
                    checkFlowProperties(sSig, dSig, oFlow, oFlowSize);

                    e2 = emd(dSig, sSig, dist_l1, rFlow, &rFlowSize);
                    checkFlowProperties(dSig, sSig, rFlow, rFlowSize);
                    shouldEqualToleranceMessage(e1, e2, errorTolerance,
                            "emd(s1, s2) != emd(s2, s1)");
                    shouldMsg(oFlowSize == rFlowSize,
                            "Flows from emd(s1, s2) and emd(s2, s1) should have the same size.");

                }
                // free stuff
                delete [] oFlow;
                delete [] rFlow;
            }
        }
    }

    // Test output of emd() on source and target signatures containing only
    // one entry.
    void testEMDRandomOneToOne()
    {
        const int nTries = 100;
        const double maxWeight = 1000;
        const int maxBinIndex = 1000;

        int sourceFeatures[1] = {0}, targetFeatures[1] = {0};
        double sourceWeights[1] = {0}, targetWeights[1] = {0};
        Signature<int> sourceSignature(sourceFeatures, sourceWeights, 1),
            targetSignature(targetFeatures, targetWeights, 1);

        double   e;
        flow_t      flow[1]; // flow size is bounded by sig1->n + sig2->n - 1
        int         flowSize;
        int sBinIndex, dBinIndex;
        double weight;

        for (int i=0; i < nTries; ++i)
        {
            sBinIndex = random.uniformInt(maxWeight);
            dBinIndex = random.uniformInt(maxWeight);
            weight = random.uniform(0., maxWeight);

            sourceSignature.features_[0] = sBinIndex;
            targetSignature.features_[0] = dBinIndex;
            sourceSignature.weights_[0] = weight;
            targetSignature.weights_[0] = weight;

            e = emd(sourceSignature, targetSignature,
                    dist_l1, flow, &flowSize);
            shouldMsg(flowSize == 1,
                    "Flow between single entry signatures should have 1 element.");
            shouldEqualToleranceMessage(e, dist_l1(sBinIndex, dBinIndex),
                    errorTolerance, "EMD between single entry signatures not in expected tolerance.");
        }
    }

    // Test output of emd() with source or target signature with single entry.
    void testEMDRandomOneMany()
    {
        // max bin sizes
        int binBounds[] = {10, 15, 20, 50, 100};
        const int nTriesPerSize = 100;
        const double maxWeight = 1000;
        const int maxBinIndex = 1000;

        int singleFeatures[1] = {0};
        double singleWeights[1] = {0};
        Signature<int> singleSignature(singleFeatures, singleWeights, 1);

        Signature<int> manySig;

        double   e;
        flow_t      *flow;
        int         flowSize;
        int sBinIndex, dBinIndex;
        double weight;
        double expected;

        int lenBinBounds = sizeof(binBounds)/sizeof(int);
        for (int i=0; i < lenBinBounds; ++i)
        {
            int cBound = binBounds[i];
            for (int j=0; j < nTriesPerSize; ++j)
            {
                sBinIndex = random.uniformInt(maxWeight);
                weight = random.uniform(0., maxWeight);
                singleSignature.features_[0] = sBinIndex;
                singleSignature.weights_[0] = weight;

                manySig.randomize(weight, cBound, random);
                flow = new flow_t[cBound];

                e = emd(singleSignature, manySig, dist_l1, flow, &flowSize);
                shouldMsg(flowSize == manySig.size(),
                        "Flow should have one arrow for each bin in the target signature.");
                checkFlowProperties(singleSignature, manySig, flow,
                        flowSize);

                // calculate expected value
                expected = 0;

                for(int k=0; k < manySig.size(); ++k)
                {
                    expected += dist_l1(manySig.features_[k], singleSignature.features_[0]) * manySig.weights_[k];
                }
                expected /= weight;

                shouldEqualToleranceMessage(e, expected, errorTolerance,
                        "emd not equal to expected value.");

                e = emd(manySig, singleSignature, dist_l1, flow, &flowSize);
                shouldMsg(flowSize == manySig.size(),
                        "Flow should have one arrow for each bin in the source signature.");
                checkFlowProperties(manySig, singleSignature, flow,
                        flowSize);
                shouldEqualToleranceMessage(e, expected, errorTolerance,
                        "emd not equal to expected value.");

                delete [] flow;
            }
        }
    }

    void testEMDRandomScaleWeights()
    {
        // max bin sizes
        int binBounds[] = {10, 15, 20, 50, 100};
        // how many random signatures should be generated for each size
        const int nTriesPerSize = 10;
        // bound for the total weight of the signatures
        const double maxWeight = 1000;

        double e1, e2, scaleFactor;
        flow_t *flow;
        int flowSize;
        Signature<int> sSig, dSig;

        int lenBinBounds = sizeof(binBounds)/sizeof(int);
        for (int i=0; i < lenBinBounds; ++i)
        {
            int sBound = binBounds[i];
            for (int j=0; j < lenBinBounds; ++j)
            {
                int dBound = binBounds[j];
                // flow size is bounded by sBound + dBound - 1
                flow = new flow_t[sBound + dBound  - 1];
                for (int k=0; k < nTriesPerSize; ++k)
                {
                    // generateRandomSignature allocates memory for sig
                    sSig.randomize(random.uniform(0., maxWeight), sBound,
                            random);
                    dSig.randomize(random.uniform(0., maxWeight), dBound,
                            random);
                    e1 = emd(sSig, dSig, dist_l1, flow, &flowSize);
                    checkFlowProperties(sSig, dSig, flow, flowSize);

                    // scale source and target weights
                    scaleFactor = random.uniform(.5, 5.);
                    sSig.scale(scaleFactor);
                    dSig.scale(scaleFactor);

                    e2 = emd(sSig, dSig, dist_l1, flow, &flowSize);
                    shouldEqualToleranceMessage(e1, e2, errorTolerance,
                            "emd should be invariant under scaling weights.");
                    checkFlowProperties(sSig, dSig, flow, flowSize);
                }
                delete [] flow;
            }
        }
    }
};



struct HistogramDistanceTestSuite : public vigra::test_suite
{
    HistogramDistanceTestSuite()
        : vigra::test_suite("HistogramDistanceTestSuite")
    {
        add(testCase(&EarthMoverDistanceTest::testEMD_RTG_example1));
        add(testCase(&EarthMoverDistanceTest::testEMD_RTG_example2));
        add(testCase(&EarthMoverDistanceTest::testEMDEpsilon));
        add(testCase(&EarthMoverDistanceTest::testEMDEmptyInOut));
        add(testCase(&EarthMoverDistanceTest::testEMDRandomToSelf));
        add(testCase(&EarthMoverDistanceTest::testEMDRandomSymmetric));
        add(testCase(&EarthMoverDistanceTest::testEMDRandomOneToOne));
        add(testCase(&EarthMoverDistanceTest::testEMDRandomOneMany));
        add(testCase(&EarthMoverDistanceTest::testEMDRandomScaleWeights));
    }
};


int main (int argc, char ** argv)
{
    HistogramDistanceTestSuite test;
    const int failed = test.run(vigra::testsToBeExecuted(argc, argv));
    std::cout << test.report() << std::endl;

    return failed != 0;
}
