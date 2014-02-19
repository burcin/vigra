/************************************************************************/
/*                                                                      */
/*        Copyright 2011-2013 by Luca Fiaschi                           */
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

#include <vigra/random_forest.hxx>
#include <vigra/random_forest_hdf5_impex.hxx>
#include <vigra/random_forest/rf_more_visitor.hxx>

#include <set>
#include <cmath>
#include <memory>
#include <limits>

#include <cstdio>
#include <boost/filesystem.hpp>


namespace vigra
{

template<class LabelType, class T =float>
class Hough_Forest: public RandomForest<LabelType, HoughTag>
{
public:
    RandomForestOptions options_;

    typedef std::pair<int, int> LeafType;
    typedef int SampleType;

    vigra::rf::visitors::IndexVisitorTraining visitor_learning;
    vigra::rf::visitors::IndexVisitorPrediction visitor_prediction;

    int dim_labels;
    int dim_features;

    int ntr_samples;
    int nte_samples;

    DepthAndSizeStopping stop;

    MultiArray<2,LabelType> tr_labels;

    Hough_Forest(RandomForestOptions options, int max_depth, int min_size) :
        RandomForest<LabelType, HoughTag> (options), options_(options),
        stop(max_depth, min_size)
    {
        initParams();
    }


    Hough_Forest(RandomForestOptions options) :
        RandomForest<LabelType, HoughTag> (options), options_(options)
    {
        initParams();
    }

    Hough_Forest(int treeCount) :
        RandomForest<LabelType, HoughTag> (
                vigra::RandomForestOptions().tree_count(treeCount))
    {
        initParams();
    }

    Hough_Forest()
    {
        initParams();
    }

    void initParams()
    {
        dim_labels = 0;
        dim_features = 0;
        ntr_samples = 0;
        nte_samples = 0;
    }

    template <class C, class Split_t, class Random_t>
    void learn(const MultiArrayView<2, LabelType, C>& train_data,
            const MultiArrayView<2, LabelType, C>& train_labels,
            Split_t split, Random_t &rng)
    {
        ntr_samples = train_data.shape(0);
        dim_labels = train_labels.shape(1);
        dim_features = train_data.shape(1);

        tr_labels = train_labels;

        rf::visitors::RandomForestProgressVisitor progressvisit;

        RandomForest<LabelType, HoughTag>::learn(train_data, train_labels,
                create_visitor((*this).visitor_learning, progressvisit),
                split, stop, rng);
    }

    template <class C, class Split_t>
    void learn(const MultiArrayView<2, LabelType, C>& train_data,
            const MultiArrayView<2, LabelType, C>& train_labels,
            Split_t split, UInt32 randomSeed)
    {
        RandomNumberGenerator<> rng(randomSeed, randomSeed == 0);
        learn(train_data, train_labels, split, rng);
    }


    template <class C, class Split_t, class Random_t>
    void learn(const MultiArrayView<2, LabelType, C>& train_data,
            const MultiArrayView<2, LabelType, C>& train_labels,
            Split_t split)
    {
        learn(train_data, train_labels, split, rf_default());
    }

    //FIXME: Random Gini and Random Entropy work in general a bit better BUT the implementation makes them much slower
    // the implementation of the split functor should be completely reworked
    template <class C>
    void learnRandomGini(const MultiArrayView<2, LabelType, C>& train_data,
            const MultiArrayView<2, LabelType, C>& train_labels,
            UInt32 randomSeed=0)
    {
        vigra::rf::split::HoughSplitRandomGini rsplit;
        learn(train_data, train_labels, rsplit, randomSeed);
    }

    template <class C>
    void learnRandomEntropy(const MultiArrayView<2, LabelType, C>& train_data,
            const MultiArrayView<2, LabelType, C>& train_labels,
            UInt32 randomSeed=0)
    {
        vigra::rf::split::HoughSplitRandomEntropy rsplit;
        learn(train_data, train_labels, rsplit, randomSeed);
    }

    template <class C>
    void learnOrthogonalEntropy(const MultiArrayView<2,LabelType,C>& train_data,
            const MultiArrayView<2, LabelType, C>& train_labels,
            UInt32 randomSeed=0)
    {
        vigra::rf::split::HoughSplitEntropy rsplit;
        learn(train_data, train_labels, rsplit, randomSeed);
    }

    template <class C>
    void learnOrthogonalGini(const MultiArrayView<2, LabelType, C>& train_data,
            const MultiArrayView<2, LabelType, C>& train_labels,
            UInt32 randomSeed=0)
    {
        vigra::rf::split::HoughSplitGini rsplit;
        learn(train_data, train_labels, rsplit, randomSeed);
    }

    template<class result, class A>
    void predict(const A& test_data, result res)
    {
        visitor_prediction.clear();

        return this->predictProbabilities(
                test_data, res, rf_default(), (*this).visitor_prediction);
    }

    /**
     * This is the function that mekes the prediction of the hough forests
     *
     * @param test_data Must be a set of patches for which we want to predict reshaped as a matrix N*nfeat (can be all patches in the image)
     * @param patch_centers (must store the centre of each patch N*2
     * @param imgwidth width of the image we want to predict
     * @param imgheight heigth of the image we want to predict
     * @param factor gain factor of the image to predict (contrast)
     * @param result *initialized result
     */
    template<class T2, class C2, class T3, class C3>
    void predictOnImage(MultiArrayView<2, T2, C2> test_data, MultiArrayView<2, T2,
            C2> patch_centers, int imgwidth, int imgheight,int factor,
            MultiArrayView<2, T3, C3> result)
    {

        typedef std::vector<double> Point;

        //FIXME: currently the algorithm assumes that result shape is imgheight,imgwidth
        // problably it is redundand to pass this two parameters from outside
        if (result.shape(0) != imgheight or result.shape(1) != imgwidth)
            throw std::runtime_error("The result has a wrong shape!, shoudl be (imgheight,imgwidth)");

        if (patch_centers.shape(0) != test_data.shape(0))
            throw std::runtime_error("shape mismatch");

        //FIXME in the far future, Generalize this function to multilabel Hough Forest
        MultiArrayShape<2>::type sh(test_data.shape(0), 2);

        MultiArray<2, T> dummy(sh);

        this->predict(test_data, dummy);


        int count = 0;
        int inscount = 0;

        //Loop on the samples

        for (int i = 0; i < test_data.shape(0); i++)
        {
            std::vector<LeafType>& leafs =
                    (this->visitor_prediction).mapping[i];

            int ox = patch_centers(i, 0); //patch centers offsets vectors
            int oy = patch_centers(i, 1);

            double nleafs = leafs.size(); // number of leafs in the associated node
            //std::cout << "N leafs " << nleafs << std::endl;


            for (std::vector<LeafType>::iterator it = leafs.begin(); it
                    != leafs.end(); it++)
            {
                std::vector<Point>& current_points =
                        (this->visitor_learning).mapping[*it];

                double prob = (this->visitor_learning).mapping_to_prob[*it];
                //std::cout<<"the prob is " << prob << std::endl;

                std::vector<Point>::iterator ii;

                double npoints = current_points.size();

                for (ii = current_points.begin(); ii != current_points.end(); ++ii)
                {

                    int x = (*ii)[0] + ox;
                    int y = (*ii)[1] + oy;

                    //std::cout<< "The offset is " <<x << " " << y << std::endl;

                    //std::cout<< " ( " << x  << ", " << y << " ) " ;

                    if ((0 <= x) & (x < imgwidth) & (0 <= y) & (y < imgheight))
                    {
                        //res(y,x)+=prob/npoints/nleafs;
                        //res(y,x)+=prob/npoints/nleafs*100;

                        //if (prob > 1)
                        //    throw std::runtime_error("prob>1");

                        //if (prob < 0)
                        //    throw std::runtime_error("prob<0");
                        //std::cerr << x  << ", ";

                        result(y, x) += factor * prob / (npoints * nleafs +1e-42);

                        inscount++;
                    }

                    //else
                    //{
                    //std::cout << "outside" << x << " " << y << std::endl;
                    //    ++count;
                    //}
                }
            }
        }
        //std::cout << "the outside vote count is " << count << std::endl;
        //std::cout << "the inside vote count is " << inscount << std::endl;
    }
};

#ifdef HasHDF5

template<class LabelType, class T>
bool hf_import_HDF5(Hough_Forest<LabelType, T> &Hf, const std::string &fname)
{
    std::string pathname = "forest";

    if(!boost::filesystem::exists(fname))
        throw std::runtime_error("File  does not exists!");

    HDF5ImportInfo info(fname.c_str(),"parameters/tr_labels");

    //FIXME could this crash??
    MultiArrayShape<2>::type shape(info.shapeOfDimension(0), info.shapeOfDimension(1));

    Hf.tr_labels.reshape(shape);
    readHDF5(info,Hf.tr_labels);

    rf_import_HDF5(Hf,fname,pathname);

    Hf.visitor_learning.load(fname,"parameters/vLearning");

    return 1;
}

template<class LabelType, class T>
void hf_saveToHDF5(const Hough_Forest<LabelType,T> &Hf,
        const std::string &fname)
{
    if(boost::filesystem::exists(fname))
        remove( fname.c_str() );

    if(!boost::filesystem::exists(fname))
        throw std::runtime_error("File exists!");

    rf_export_HDF5(Hf, fname, "forest");

    Hf.visitor_learning.save(fname,"parameters/vLearning");

    writeHDF5(fname.c_str(),"parameters/tr_labels",Hf.tr_labels);
}

#endif // HasHDF5

} // namespace vigra


