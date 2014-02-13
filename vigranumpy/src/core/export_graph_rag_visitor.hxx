#ifndef VIGRA_EXPORT_GRAPH_RAG_VISITOR_HXX
#define VIGRA_EXPORT_GRAPH_RAG_VISITOR_HXX
//#define NO_IMPORT_ARRAY

/*std*/
#include <sstream>
#include <string>

/*vigra*/
#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>
#include <boost/python.hpp>
#include <vigra/graphs.hxx>
#include <vigra/graph_helper/dense_map.hxx>
#include <vigra/graph_helper/on_the_fly_edge_map.hxx>
#include <vigra/python_graph_generalization.hxx>
#include <vigra/graph_algorithms.hxx>
#include <vigra/metrics.hxx>
#include <vigra/multi_gridgraph.hxx>
#include <vigra/error.hxx>
namespace python = boost::python;

namespace vigra{








template<class GRAPH>
class LemonGraphRagVisitor 
:   public boost::python::def_visitor<LemonGraphRagVisitor<GRAPH> >
{
public:

    friend class def_visitor_access;

    typedef GRAPH Graph;
    typedef AdjacencyListGraph RagGraph;

    typedef LemonGraphRagVisitor<GRAPH> VisitorType;
    // Lemon Graph Typedefs
    
    typedef typename Graph::index_type       index_type;
    typedef typename Graph::Edge             Edge;
    typedef typename Graph::Node             Node;
    typedef typename Graph::Arc              Arc;
    typedef typename Graph::NodeIt           NodeIt;
    typedef typename Graph::EdgeIt           EdgeIt;
    typedef typename Graph::ArcIt            ArcIt;

    typedef typename RagGraph::Edge             RagEdge;
    typedef typename RagGraph::Node             RagNode;
    typedef typename RagGraph::Arc              RagArc;
    typedef typename RagGraph::NodeIt           RagNodeIt;
    typedef typename RagGraph::EdgeIt           RagEdgeIt;
    typedef typename RagGraph::ArcIt            RagArcIt;


    typedef EdgeHolder<Graph> PyEdge;
    typedef NodeHolder<Graph> PyNode;
    typedef  ArcHolder<Graph> PyArc;


    // predefined array (for map usage)
    const static unsigned int EdgeMapDim = IntrinsicGraphShape<Graph>::IntrinsicEdgeMapDimension;
    const static unsigned int NodeMapDim = IntrinsicGraphShape<Graph>::IntrinsicNodeMapDimension;

    typedef NumpyArray<EdgeMapDim,   Singleband<float > > FloatEdgeArray;
    typedef NumpyArray<NodeMapDim,   Singleband<float > > FloatNodeArray;
    typedef NumpyArray<NodeMapDim,   Singleband<UInt32> > UInt32NodeArray;
    typedef NumpyArray<NodeMapDim,   Singleband<Int32 > > Int32NodeArray;
    typedef NumpyArray<NodeMapDim +1,Multiband <float > > MultiFloatNodeArray;

    typedef NumpyScalarEdgeMap<Graph,FloatEdgeArray>         FloatEdgeArrayMap;
    typedef NumpyScalarNodeMap<Graph,FloatNodeArray>         FloatNodeArrayMap;
    typedef NumpyScalarNodeMap<Graph,UInt32NodeArray>        UInt32NodeArrayMap;
    typedef NumpyScalarNodeMap<Graph,Int32NodeArray>         Int32NodeArrayMap;
    typedef NumpyMultibandNodeMap<Graph,MultiFloatNodeArray> MultiFloatNodeArrayMap;


   
    const static unsigned int RagEdgeMapDim = IntrinsicGraphShape<RagGraph>::IntrinsicEdgeMapDimension;
    const static unsigned int RagNodeMapDim = IntrinsicGraphShape<RagGraph>::IntrinsicNodeMapDimension;

    typedef NumpyArray<RagEdgeMapDim,   Singleband<float > > RagFloatEdgeArray;
    typedef NumpyArray<RagNodeMapDim,   Singleband<float > > RagFloatNodeArray;
    typedef NumpyArray<RagNodeMapDim,   Singleband<UInt32> > RagUInt32NodeArray;
    typedef NumpyArray<RagNodeMapDim,   Singleband<Int32 > > RagInt32NodeArray;
    typedef NumpyArray<RagNodeMapDim +1,Multiband <float > > RagMultiFloatNodeArray;

    typedef NumpyScalarEdgeMap<RagGraph,RagFloatEdgeArray>         RagFloatEdgeArrayMap;
    typedef NumpyScalarNodeMap<RagGraph,RagFloatNodeArray>         RagFloatNodeArrayMap;
    typedef NumpyScalarNodeMap<RagGraph,RagUInt32NodeArray>        RagUInt32NodeArrayMap;
    typedef NumpyScalarNodeMap<RagGraph,RagInt32NodeArray>         RagInt32NodeArrayMap;
    typedef NumpyMultibandNodeMap<RagGraph,RagMultiFloatNodeArray> RagMultiFloatNodeArrayMap;



    typedef typename RagGraph:: template EdgeMap< std::vector<Edge> > RagAffiliatedEdges;


    typedef typename GraphDescriptorToMultiArrayIndex<Graph>::IntrinsicNodeMapShape NodeCoordinate;
    typedef NumpyArray<1,NodeCoordinate>  NodeCoorinateArray;

    LemonGraphRagVisitor(const std::string clsName)
    :clsName_(clsName){

    }

    void exportRagAffiliatedEdges()const{

        const std::string hyperEdgeMapNamClsName = clsName_ + std::string("RagAffiliatedEdges");
        python::class_<RagAffiliatedEdges>(hyperEdgeMapNamClsName.c_str(),python::init<const RagGraph &>())
        ;

    }

    template <class classT>
    void visit(classT& c) const
    {   

        // something like RagEdgeMap< std::vector< Edge > >
        exportRagAffiliatedEdges();

        // make the region adjacency graph
        python::def("_regionAdjacencyGraph",registerConverters(&pyMakeRegionAdjacencyGraph),
            python::return_value_policy<  python::manage_new_object >()
        );

        python::def("_ragEdgeFeatures",registerConverters(&pyRagEdgeFeatures),
            (
                python::arg("rag"),
                python::arg("graph"),
                python::arg("affiliatedEdges"),
                python::arg("edgeFeatures"),
                python::arg("acc"),
                python::arg("out")=python::object()
            )
        );

        python::def("_ragNodeFeatures",registerConverters(&pyRagNodeFeaturesMultiband),
            (
                python::arg("rag"),
                python::arg("graph"),
                python::arg("labels"),
                python::arg("nodeFeatures"),
                python::arg("acc"),
                python::arg("ignoreLabel")=-1,
                python::arg("out")=python::object()
            )
        );
        python::def("_ragNodeFeatures",registerConverters(&pyRagNodeFeaturesSingleband),
            (
                python::arg("rag"),
                python::arg("graph"),
                python::arg("labels"),
                python::arg("nodeFeatures"),
                python::arg("acc"),
                python::arg("ignoreLabel")=-1,
                python::arg("out")=python::object()
            )
        );

        python::def("_ragNodeSize",registerConverters(&pyRagNodeSize),
            (
                python::arg("rag"),
                python::arg("graph"),
                python::arg("labels"),
                python::arg("ignoreLabel")=-1,
                python::arg("out")=python::object()
            )
        );
        python::def("_ragEdgeSize",registerConverters(&pyRagEdgeSize),
            (
                python::arg("rag"),
                python::arg("affiliatedEdges"),
                python::arg("out")=python::object()
            )
        );
    }


    static RagAffiliatedEdges * pyMakeRegionAdjacencyGraph(
        const Graph &   graph,
        UInt32NodeArray labelsArray,
        RagGraph &      rag,
        const Int32 ignoreLabel=-1
    ){
        // numpy arrays => lemon maps
        UInt32NodeArrayMap labelsArrayMap(graph,labelsArray);

        // allocate a new RagAffiliatedEdges
        RagAffiliatedEdges * affiliatedEdges = new RagAffiliatedEdges(rag);

        // call algorithm itself
        makeRegionAdjacencyGraph(graph,labelsArrayMap,rag,*affiliatedEdges,ignoreLabel);

        return affiliatedEdges;
    }

    static NumpyAnyArray  pyRagEdgeFeatures(
        const RagGraph &           rag,
        const Graph &              graph,
        const RagAffiliatedEdges & affiliatedEdges,
        FloatEdgeArray             edgeFeaturesArray,
        const std::string &        accumulator,
        RagFloatEdgeArray          ragEdgeFeaturesArray
    ){

        vigra_precondition(accumulator==std::string("mean") || accumulator==std::string("sum"),
            "currently the accumulators are limited to mean and sum"
        );

        // resize out
        ragEdgeFeaturesArray.reshapeIfEmpty(IntrinsicGraphShape<RagGraph>::intrinsicEdgeMapShape(rag));
        std::fill(ragEdgeFeaturesArray.begin(),ragEdgeFeaturesArray.end(),0.0f);
        // numpy arrays => lemon maps
        FloatEdgeArrayMap    edgeFeaturesArrayMap(graph,edgeFeaturesArray);
        RagFloatEdgeArrayMap ragEdgeFeaturesArrayMap(rag,ragEdgeFeaturesArray);

        const bool isMeanAcc= accumulator==std::string("mean");
        for(RagEdgeIt iter(rag);iter!=lemon::INVALID;++iter){
            const RagEdge ragEdge = *iter;
            const std::vector<Edge> & affEdges = affiliatedEdges[ragEdge];
            for(size_t i=0;i<affEdges.size();++i){
                ragEdgeFeaturesArrayMap[ragEdge]+=edgeFeaturesArrayMap[affEdges[i]];
            }
            if(isMeanAcc){
                ragEdgeFeaturesArrayMap[ragEdge]/=static_cast<float>(affEdges.size());
            }
        }
        return ragEdgeFeaturesArray;
    }


    static NumpyAnyArray  pyRagNodeFeaturesSingleband(
        const RagGraph &           rag,
        const Graph &              graph,
        UInt32NodeArray            labelsArray,
        FloatNodeArray             nodeFeaturesArray,
        const std::string &        accumulator,
        const Int32                ignoreLabel=-1,
        RagFloatNodeArray          ragNodeFeaturesArray=RagFloatNodeArray()
    ){

        vigra_precondition(accumulator==std::string("mean") || accumulator==std::string("sum"),
            "currently the accumulators are limited to mean and sum"
        );

        // resize out

        ragNodeFeaturesArray.reshapeIfEmpty(IntrinsicGraphShape<RagGraph>::intrinsicNodeMapShape(rag));
        std::fill(ragNodeFeaturesArray.begin(),ragNodeFeaturesArray.end(),0.0f);

        // numpy arrays => lemon maps
        UInt32NodeArrayMap   labelsArrayMap(graph,labelsArray);
        FloatNodeArrayMap    nodeFeaturesArrayMap(graph,nodeFeaturesArray);
        RagFloatNodeArrayMap ragNodeFeaturesArrayMap(rag,ragNodeFeaturesArray);

        if(accumulator == std::string("mean")){
            typename RagGraph:: template NodeMap<float> counting(rag,0.0f);
            for(NodeIt iter(graph);iter!=lemon::INVALID;++iter){
                UInt32 l = labelsArrayMap[*iter];
                if(ignoreLabel==-1 || static_cast<Int32>(l)!=ignoreLabel){
                    const RagNode ragNode   = rag.nodeFromId(l);
                    ragNodeFeaturesArrayMap[ragNode]+=nodeFeaturesArrayMap[*iter];
                    counting[ragNode]+=1.0;
                }
            }
            for(RagNodeIt iter(rag);iter!=lemon::INVALID;++iter){
                const RagNode ragNode   = *iter;
                ragNodeFeaturesArrayMap[ragNode]/=counting[ragNode];
            }
        }
        else{
            for(NodeIt iter(graph);iter!=lemon::INVALID;++iter){
                UInt32 l = labelsArrayMap[*iter];
                if(ignoreLabel==-1 || static_cast<Int32>(l)!=ignoreLabel){
                    const RagNode ragNode   = rag.nodeFromId(l);
                    ragNodeFeaturesArrayMap[ragNode]+=nodeFeaturesArrayMap[*iter];
                }
            }
        }
        return ragNodeFeaturesArray;
    }


    static NumpyAnyArray  pyRagNodeFeaturesMultiband(
        const RagGraph &           rag,
        const Graph &              graph,
        UInt32NodeArray            labelsArray,
        MultiFloatNodeArray        nodeFeaturesArray,
        const std::string &        accumulator,
        const Int32                ignoreLabel=-1,
        RagMultiFloatNodeArray     ragNodeFeaturesArray=RagMultiFloatNodeArray()
    ){
        vigra_precondition(accumulator==std::string("mean") || accumulator==std::string("sum"),
            "currently the accumulators are limited to mean and sum"
        );

        // resize out
        typename MultiArray<RagNodeMapDim+1,int>::difference_type outShape;
        for(size_t d=0;d<RagNodeMapDim;++d){
            outShape[d]=IntrinsicGraphShape<RagGraph>::intrinsicNodeMapShape(rag)[d];
        }
        outShape[RagNodeMapDim]=nodeFeaturesArray.shape(NodeMapDim);

        ragNodeFeaturesArray.reshapeIfEmpty(   RagMultiFloatNodeArray::ArrayTraits::taggedShape(outShape,"xc") );
        std::fill(ragNodeFeaturesArray.begin(),ragNodeFeaturesArray.end(),0.0f);

        // numpy arrays => lemon maps
        UInt32NodeArrayMap        labelsArrayMap(graph,labelsArray);
        MultiFloatNodeArrayMap    nodeFeaturesArrayMap(graph,nodeFeaturesArray);
        RagMultiFloatNodeArrayMap ragNodeFeaturesArrayMap(rag,ragNodeFeaturesArray);

        if(accumulator == std::string("mean")){
            typename RagGraph:: template NodeMap<float> counting(rag,0.0f);
            for(NodeIt iter(graph);iter!=lemon::INVALID;++iter){
                UInt32 l = labelsArrayMap[*iter];
                if(ignoreLabel==-1 || static_cast<Int32>(l)!=ignoreLabel){
                    const RagNode ragNode   = rag.nodeFromId(l);
                    ragNodeFeaturesArrayMap[ragNode]+=nodeFeaturesArrayMap[*iter];
                    counting[ragNode]+=1.0;
                }
            }
            for(RagNodeIt iter(rag);iter!=lemon::INVALID;++iter){
                const RagNode ragNode   = *iter;
                ragNodeFeaturesArrayMap[ragNode]/=counting[ragNode];
            }
        }
        else{
            for(NodeIt iter(graph);iter!=lemon::INVALID;++iter){
                UInt32 l = labelsArrayMap[*iter];
                if(ignoreLabel==-1 || static_cast<Int32>(l)!=ignoreLabel){
                    const RagNode ragNode   = rag.nodeFromId(l);
                    ragNodeFeaturesArrayMap[ragNode]+=nodeFeaturesArrayMap[*iter];
                }
            }
        }
        return ragNodeFeaturesArray;
    }

    static NumpyAnyArray  pyRagNodeSize(
        const RagGraph &           rag,
        const Graph &              graph,
        UInt32NodeArray            labelsArray,
        const Int32                ignoreLabel=-1,
        RagFloatNodeArray          ragNodeSizeArray=RagFloatNodeArray()
    ){
        // resize out
        ragNodeSizeArray.reshapeIfEmpty(IntrinsicGraphShape<RagGraph>::intrinsicNodeMapShape(rag));
        std::fill(ragNodeSizeArray.begin(),ragNodeSizeArray.end(),0.0f);

        // numpy arrays => lemon maps
        UInt32NodeArrayMap labelsArrayMap(graph,labelsArray);
        RagFloatNodeArrayMap ragNodeSizeArrayMap(rag,ragNodeSizeArray);
        for(NodeIt iter(graph);iter!=lemon::INVALID;++iter){
            UInt32 l = labelsArrayMap[*iter];
            if(ignoreLabel==-1 || static_cast<Int32>(l)!=ignoreLabel){
                const RagNode ragNode   = rag.nodeFromId(l);
                ragNodeSizeArrayMap[ragNode]+=1.0f;
            }
        }

        return ragNodeSizeArray;
    }

    static NumpyAnyArray  pyRagEdgeSize(
        const RagGraph &           rag,
        const RagAffiliatedEdges & affiliatedEdges,
        RagFloatEdgeArray          ragEdgeFeaturesArray
    ){
        // reshape out
        ragEdgeFeaturesArray.reshapeIfEmpty(IntrinsicGraphShape<RagGraph>::intrinsicEdgeMapShape(rag));
        // numpy arrays => lemon maps
        RagFloatEdgeArrayMap ragEdgeFeaturesArrayMap(rag,ragEdgeFeaturesArray);

        for(RagEdgeIt iter(rag);iter!=lemon::INVALID;++iter){
            const RagEdge ragEdge = *iter;
            const std::vector<Edge> & affEdges = affiliatedEdges[ragEdge];
            ragEdgeFeaturesArrayMap[ragEdge]=static_cast<float>(affEdges.size());
        }
        return ragEdgeFeaturesArray;
    }




private:
    std::string clsName_;
};




} // end namespace vigra

#endif // VIGRA_EXPORT_GRAPH_RAG_VISITOR_HXX