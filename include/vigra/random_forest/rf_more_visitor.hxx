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

#ifndef RF_MORE_VISITORS_HXX
#define RF_MORE_VISITORS_HXX

#include <map>

#include "rf_visitors.hxx"

namespace vigra
{
namespace rf
{
namespace visitors
{



/** Visitor that calculates different OOB error statistics
 */
class OOB_VariancePerLeaf : public VisitorBase
{
public:
    typedef int tree_index;
    typedef int leaf_index;
    typedef int example_index; // index of the sample in the feature matrix

    typedef std::vector<double> Point;
    typedef std::pair<tree_index, leaf_index> LeafType;
    typedef std::map<LeafType, std::vector<
            example_index> >::iterator KeyIt;
    typedef std::vector<example_index>::iterator ValueIt;

    std::map<LeafType, double> mapping_to_count; // map the tree and the leaf to the average in the leaf
    std::map<LeafType, Point> mapping_to_variance; // map the tree and the leaf to the average in the leaf

    double totalVariance;


    OOB_VariancePerLeaf(): totalVariance(0){};

    template<class RF, class PR, class SM, class ST>
    void visit_after_tree(RF& rf, PR & pr,  SM & sm, ST & st, int index)
    {
        int nlabels=pr.response().shape(1);
        // go through the samples
        for(int ll = 0; ll < rf.ext_param_.row_count_; ++ll)
        {
            // if the lth sample is oob...
            //if(!sm.is_used()[ll])
            {
                // update number of trees in which current sample is oob
                // get the predicted votes ---> tmp_prob;
                ArrayVector<double>::const_iterator response;
                int pos =  rf.tree(index).getToLeaf(rowVector(pr.features(),ll));
                response=rf.trees_[index].predict(rowVector(pr.features(),ll));
                LeafType leaf(index,pos);

                if (!mapping_to_variance.count(leaf))
                    {
                        Point & sum=mapping_to_variance[leaf];
                        sum.resize(rf.ext_param_.class_count_);
                        for(int k=0; k<nlabels; ++k)
                            sum[k]=0;
                        double & nsamples = mapping_to_count[leaf];
                        nsamples=0;
                    }

                double & nsamples = mapping_to_count[leaf];
                ++nsamples;

                Point & sum=mapping_to_variance[leaf];
                for(int k=0; k<nlabels; ++k)
                {
//std::cerr << "sample index " << ll << "  prediction " << response[k] << " true " << rowVector(pr.response(),ll)[k] << std::endl;
                    sum[k]+=(response[k]-rowVector(pr.response(),ll)[k])*(response[k]-rowVector(pr.response(),ll)[k]);
                }
            }
        }
    }

    /** Normalise variable importance after the number of trees is known.
     */
    template<class RF, class PR>
    void visit_at_end(RF & rf, PR & pr)
    {
        //std::cerr << "NORMALIZATIOOOOOONNNN " << std::endl;
        //normalize

        Point total(rf.ext_param_.class_count_,0);

        int count=0;
        for (std::map<LeafType, Point >::iterator it=mapping_to_variance.begin();it!=mapping_to_variance.end();it++)
        {    ++count;
            Point & sum = mapping_to_variance[(*it).first];
            double & nsamples = mapping_to_count[(*it).first];

            for(int k=0; k<rf.ext_param_.class_count_; ++k)
            {
                sum[k]/=nsamples;
                total[k]+=sum[k];
            }
        }


        for(int k=0; k<rf.ext_param_.class_count_; ++k)
            {
                total[k]/=count;
                this->totalVariance+=total[k];
            }

        std::cerr << "NORMALIZATIOOOOOONNNN " << this->totalVariance << std::endl;
    }
};


/** This visitor is used at prediction time to recall where the current
 * sample ends in the leaf and then cast the votes for the hough forest
 */
class IndexVisitorPrediction: public VisitorBase
{

public:
    typedef int tree_index;
    typedef int leaf_index;
    typedef int example_index; // index of the sample in the feature matrix

    typedef std::map<example_index, std::vector<std::pair<tree_index,
            leaf_index> > >::iterator KeyIt;
    typedef std::vector<std::pair<tree_index, leaf_index> >::iterator ValueIt;

    int current_tree;
    int current_sample;
    std::map<example_index, std::vector<std::pair<tree_index, leaf_index> > >
            mapping; //map tree and leaf with index of the sample

    IndexVisitorPrediction() :
        current_tree(0), current_sample(0)
    {
    }

    /** simply increase the tree count
     */
    template<class RF, class PR, class SM, class ST>
    void visit_after_tree(RF& rf, PR & pr, SM & sm, ST & st, int index)
    {
        ++index;
        current_tree = index;
    }

    template<class RF>
    void visit_before_tree_prediction(RF& rf, int index, int sample_index)
    {
        current_sample = sample_index;
        current_tree = index;
    }

    template<class TR, class IntT, class TopT, class Feat>
    void visit_external_node(TR & tr, IntT index, TopT node_t, Feat & features)
    {
        //std::cout << "current tree " << current_tree << std::endl;
        //std::cout << "leaf address " << index << std::endl;
        //std::cout << "current samples " << current_sample << std::endl;

        //mapping[std::make_pair(current_tree, index)] = std::vector<example_index> () ;
        std::vector<std::pair<tree_index, leaf_index> > & ref =
                mapping[current_sample];

        ref.push_back(std::make_pair(current_tree, index));

        //    ++current_sample;
    }

    void save(std::string filen, std::string pathn)
    {
        //FIXME This might be slow;
        ArrayVector<int> temp;

        for (KeyIt it = mapping.begin(); it != mapping.end(); it++)
        {
            int sample = ((*it).first);

            int arr_size = ((*it).second).size();
            //std::cout << "tree " << tree << std::endl;
            temp.push_back(sample);

            temp.push_back(arr_size);
            //std::cout << "leaf " << leaf << std::endl;

            //std::cout << "arr-size " << arr_size << std::endl;

            std::vector<std::pair<tree_index, leaf_index> >& temp_vec =
                    (*it).second;

            for (int ii = 0; ii < temp_vec.size(); ii++)
            {
                int tree = temp_vec[ii].first;
                int leaf = temp_vec[ii].second;
                temp.push_back(tree);
                temp.push_back(leaf);
            }
        }

        //std::cout << temp << std::endl;

        //for(int i=0; i<temp.size();i++)
        //    std::cerr<<temp[i]<<",";
        //std::cerr<<std::endl;

        const char* filename = filen.c_str();

        MultiArrayShape<1>::type shp(temp.size());

        //FIXME this actually copy the data
        MultiArray<1u, int> multi(shp);
        multi.reshape(shp);

        for (int i = 0; i < multi.shape(0); i++)
        {
            multi(i) = temp[i];
        }

        writeHDF5(filename, pathn.c_str(), multi);
    }

    void load(std::string filen, std::string pathn)
    {
        HDF5ImportInfo info(filen.c_str(), pathn.c_str());
        vigra_precondition(info.numDimensions() == 1, "Dataset must be 1-dimensional.");

        MultiArrayShape<1>::type shp(info.shape()[0]);
        MultiArray<1, int> array(shp);

        readHDF5(info, array);

        //for (int i=0; i<shp[0];i++)
        //    std::cerr << array(i) << ",";
        //std::cerr<< std::endl;
        //std::cerr<< "array " << array << std::endl;
        mapping.clear();

        for (int it = 0; it < array.shape(0);)
        {
            int current_sample = array(it);
            ++it;

            int arr_size = array(it);
            ++it;

            std::vector<std::pair<tree_index, leaf_index> > & ref =
                    mapping[current_sample];

            //std::cerr << "arr size   " << arr_size << std::endl;

            ref.resize(arr_size);
            int i = 0;
            int current_it = it;
            while (i < arr_size)
            {
                ref[i] = std::make_pair(array(it), array(it + 1));

                ++i;
                it = it + 2;
            }
        }
    }

    void deepFout()
    {
        for(KeyIt it=mapping.begin(); it!=mapping.end();++it)
        {
            std::cerr << "sample " << (*it).first << std::endl;

            for (ValueIt ii=((*it).second).begin();ii!=((*it).second).end();++ii)
            {
             std::cerr << "(  " << (*ii).first << ", " << (*ii).second << ") " ;
            }
            std::cerr << std::endl;
        }
    }

    void clear()
    {
        current_tree = 0;
        current_sample = 0;
        mapping.clear();
    }

    int getNLeafs()
    {
        return mapping.size();
    }
};


class IndexVisitorTraining: public VisitorBase
{
    /*Visitor Used into the Hough Forest during training time*/

public:
    typedef int tree_index;
    typedef int leaf_index;
    typedef int example_index; // index of the samples in the feature matrix
    typedef std::vector<double> Point; //structure for the point in the matrix includes all the labels with the continuous regression variables

    typedef std::map<std::pair<tree_index, leaf_index>, std::vector<Point> >::iterator
            KeyIt;
    typedef std::vector<Point>::iterator ValueIt;

    int current_tree;
    int current_sample;
    int dim; //This is the size of the space of parameter the usually should be the label matrix -1 column for the categorical variables

    //FIXME: HERE we should use a better method for mapping between leafs and points ... I'm not sure what is the most convenient method
    std::map<std::pair<tree_index, leaf_index>, std::vector<Point> > mapping; //map tree and leaf with index of the sample
    std::map<std::pair<tree_index, leaf_index>, double> mapping_to_prob;

    IndexVisitorTraining(int dim_=2) :
        current_tree(0), current_sample(0), dim(dim_)
    {
    }

    /** do something after the the Split has decided how to process the Region
     * (Stack entry)
     *
     * \param tree      reference to the tree that is currently being learned
     * \param split     reference to the split object
     * \param parent    current stack entry  which was used to decide the split
     * \param leftChild left stack entry that will be pushed
     * \param rightChild
     *                  right stack entry that will be pushed.
     * \param features  features matrix
     * \param labels    label matrix
     * \sa RF_Traits::StackEntry_t
     */
    template<class Tree, class Split, class Region, class Feature_t,
            class Label_t>
    void visit_after_split(Tree & tree, Split & split, Region & parent,
            Region & leftChild, Region & rightChild, Feature_t & features,
            Label_t & labels)
    {
            leaf_index leaf_addr = tree.topology_.size();
            //std::cout << "node address " << leaf_addr << std::endl;
            //std::cout << "current tree " << current_tree << std::endl;
            //std::cout << "n samples " << parent.end()- parent.begin() << std::endl;
        if (tree.isLeafNode(split.createNode().typeID()))
        {
            //the leaf has just been created so the size of the topology gives its address
            if (mapping.find(std::make_pair(current_tree, leaf_addr))
                    != mapping.end())
                throw std::runtime_error(
                        "Visitor(): Same leaf encountered twice during learning");

            //mapping[std::make_pair(current_tree, leaf_addr)] = std::vector<Point> () ;

            std::vector<Point> & ref = mapping[std::make_pair(current_tree,
                    leaf_addr)];
            double & prob = mapping_to_prob[std::make_pair(current_tree,
                    leaf_addr)];

            ref.resize(parent.size());

            typename Region::IndexIterator it = parent.begin();

            double count_pos = 0.0;
            int k = 0;
            while (it != parent.end())
            {

                if (labels(*it, 0) != 0)
                {

                    ++count_pos;

                    //Exclude the first column that was used to define foreground background
                    Point p(labels.shape(1) - 1, 0.0);

                    //Store the information only for the column in the matrix which have label different from 0
                    for (int i = 1; i < labels.shape(1); i++)
                    {
                        if ((rowVector(labels, *it))[0] != 0)
                            p[i - 1] = ((rowVector(labels, *it))[i]);
                            if (i==3)
                                std::cerr << "The angle for the sample is " << (rowVector(labels, *it))[i] << std::endl;

                    }

                    ref[k] = p;
                    ++k;
                }

                ++it;

            }
            ref.resize(count_pos);
            prob = count_pos / (double(parent.size())+1e-32);

            //std::copy(parent.begin(), parent.end(), std::back_inserter(ref));
            // std::copy(parent.begin(), parent.end(), ref.begin());
        }
        else
        {
            //std::cout << node_addr << std::endl;
        }
    }

    /** simply increase the tree count
     */
    template<class RF, class PR, class SM, class ST>
    void visit_after_tree(RF& rf, PR & pr, SM & sm, ST & st, int index)
    {
        current_tree = index + 1;
    }

    template<class RF>
    void visit_before_tree_prediction(RF& rf, int index, int sample_index)
    {
        current_sample = sample_index;
        current_tree = index;
    }

    void save(std::string filen, std::string pathn)
    {
        //FIXME This might be slow save the mapping as a single array
        ArrayVector<double> temp;
        for (KeyIt it = mapping.begin(); it != mapping.end(); it++)
        {
            //we put treee and leaf indeces first
            int current_tree = ((*it).first).first;
            temp.push_back(current_tree);

            int current_leaf = ((*it).first).second;
            temp.push_back(current_leaf);


            // then we push back
            int arr_size = ((*it).second).size();
            temp.push_back(arr_size);

            temp.push_back(dim); //this tells how man parametes were in the three

            double prob = mapping_to_prob[(*it).first];
            temp.push_back(prob);


            //std::cout << "tree " << tree << std::endl;

            //std::cout << "leaf " << leaf << std::endl;


            //std::cout << "arr-size " << arr_size << std::endl;
            const std::vector<Point>& temp_vec=(*it).second;

            for (int i = 0; i < temp_vec.size(); i++)
                temp.insert(temp.end(), temp_vec[i].begin(), temp_vec[i].end());
        }

        //std::cout << temp << std::endl;


        //FIXME: probably in the new version of vigra is possible to avoid this double copy or view the vector as multiarray
        //SAVE to the hdf5
        const char* filename = filen.c_str();
        MultiArrayShape<1>::type shp(temp.size());
        //FIXME this actually copy the data
        MultiArray<1u, double> multi(shp);
        //multi.reshape(shp);

        for (int i = 0; i < multi.shape(0); i++)
        {
            multi(i) = temp[i];
        }

        writeHDF5(filename, pathn.c_str(), multi);
    }

    void load(std::string filen, std::string pathn)
    {
        HDF5ImportInfo info(filen.c_str(), pathn.c_str());
        vigra_precondition(info.numDimensions() == 1, "Dataset must be 1-dimensional.");

        MultiArrayShape<1>::type shp(info.shape()[0]);
        MultiArray<1, double> array(shp);

        readHDF5(info, array);

        mapping.clear();

        for (int it = 0; it < array.shape(0);)
        {
            int current_tree = array(it);
            ++it;

            int current_leaf = array(it);
            ++it;

            int arr_size = array(it);
            ++it;

            int dim = array(it);
            ++it;

            double & pr = mapping_to_prob[std::make_pair(current_tree,
                    current_leaf)];

            pr = array(it);
            ++it;

            std::vector<Point> & ref = mapping[std::make_pair(current_tree,
                    current_leaf)];

            ref.resize(arr_size);
            int i = 0;
            int current_it = it;
            while (it < current_it + arr_size * dim)
            {
                for (int k = 0; k < dim; k++)
                {
                    ref[i].push_back(array(it));
                    ++it;
                }

                ++i;
            }
        }
    }

    void clear()
    {
        current_tree = 0;
        current_sample = 0;
        mapping.clear();
        mapping_to_prob.clear();
    }

    int getNLeafs()
    {
        return mapping.size();
    }
};

class IndexVisitorTrainingRegressionForest: public VisitorBase
{
    /*Visitor Used into the Hough Forest during training time*/

public:
    typedef int tree_index;
    typedef int leaf_index;
    typedef int example_index; // index of the samples in the feature matrix
    typedef std::vector<double> Point; //structure for the point in the matrix includes all the labels with the continuous regression variables

    typedef std::map<std::pair<tree_index, leaf_index>, std::vector<Point> >::iterator
            KeyIt;
    typedef std::vector<Point>::iterator ValueIt;

    int current_tree;
    int current_sample;
    int dim; //This is the size of the space of parameter the usually should be the label matrix -1 column for the categorical variables
    std::map<std::pair<tree_index, leaf_index>, std::vector<Point> > mapping; //map tree and leaf with index of the sample in the leaf node
    std::map<std::pair<tree_index, leaf_index>, double> mapping_to_prob;

    std::map<std::pair<tree_index, leaf_index>, Point> mapping_to_average; // map the tree and the leaf to the average in the leaf
    std::map<std::pair<tree_index, leaf_index>, Point> mapping_to_variance; // map the tree and the leaf to the variance in the leaf
    std::map<std::pair<tree_index, leaf_index>, double> mapping_to_nsamples; //  map the tree and the leaf to the number of points  in the leaf


    IndexVisitorTrainingRegressionForest(int dim_=2 ) :
        current_tree(0), current_sample(0), dim(dim_)
    {
    }

    /** do something after the the Split has decided how to process the Region
     * (Stack entry)
     *
     * \param tree      reference to the tree that is currently being learned
     * \param split     reference to the split object
     * \param parent    current stack entry  which was used to decide the split
     * \param leftChild left stack entry that will be pushed
     * \param rightChild
     *                  right stack entry that will be pushed.
     * \param features  features matrix
     * \param labels    label matrix
     * \sa RF_Traits::StackEntry_t
     */
    template<class Tree, class Split, class Region, class Feature_t,
            class Label_t>
    void visit_after_split(Tree & tree, Split & split, Region & parent,
            Region & leftChild, Region & rightChild, Feature_t & features,
            Label_t & labels)
    {
            leaf_index leaf_addr = tree.topology_.size();
            //std::cout << "node address " << leaf_addr << std::endl;
            //std::cout << "current tree " << current_tree << std::endl;
            //std::cout << "n samples " << parent.end()- parent.begin() << std::endl;
        if (tree.isLeafNode(split.createNode().typeID()))
        {

            //the leaf has just been created so the size of the topology gives its address
            if (mapping.find(std::make_pair(current_tree, leaf_addr))
                    != mapping.end())
            throw std::runtime_error(
                        "Visitor(): Same leaf encountered twice during learning");


            std::vector<Point> & ref = mapping[std::make_pair(current_tree,
                    leaf_addr)];

            Point & refAverage  = mapping_to_average[std::make_pair(current_tree,
                    leaf_addr)];

            Point & refVariance  = mapping_to_variance[std::make_pair(current_tree,
                        leaf_addr)];

            double & nsamples = mapping_to_nsamples[std::make_pair(current_tree,
                    leaf_addr)];

            //init the leaf node stored data
            ref.resize(parent.size());
            refAverage.resize(labels.shape(1));
            refVariance.resize(labels.shape(1));

            for (int i = 0; i < labels.shape(1); i++)
            {refAverage[i]=0; refVariance[i]=0;    }



            int k = 0;
            typename Region::IndexIterator it;
            for (it= parent.begin();it != parent.end();it++)
            {
                Point p(labels.shape(1) , 0.0);
                    for (int i = 0; i < labels.shape(1); i++)
                    {
                        p[i] = ((rowVector(labels, *it))[i]);
                        refAverage[i]+=p[i];
                        refVariance[i]+=p[i]*p[i];
                    }
                    ref[k] = p;
                    ++k;
            }

            for (int i = 0; i < labels.shape(1); i++)
            {
                refAverage[i]/=k;
                refVariance[i]=refVariance[i]/k-refAverage[i]*refAverage[i];
            }
            nsamples=k;

        }

    }

    /** simply increase the tree count
     */
    template<class RF, class PR, class SM, class ST>
    void visit_after_tree(RF& rf, PR & pr, SM & sm, ST & st, int index)
    {
        current_tree = index + 1;
    }

    template<class RF>
    void visit_before_tree_prediction(RF& rf, int index, int sample_index)
    {
        current_sample = sample_index;
        current_tree = index;

    }

    void clear()
    {
        current_tree = 0;
        current_sample = 0;
        mapping.clear();
        mapping_to_prob.clear();
    }

    int getNLeafs()
    {
        return mapping.size();
    }
};

} //namespace visitors
} //namespace rf
} //namespace vigra
#endif
