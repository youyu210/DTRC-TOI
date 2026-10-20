std::string cipc_case;
std::vector<int> cipc_roles;
gipc::Json cipc_components = gipc::Json::array();

void set_case8()
{
    cipc_case=std::getenv("CIPC_SCENE")?std::getenv("CIPC_SCENE"):"sphere7";
    if(cipc_case!="sphere7"&&cipc_case!="bunny"&&cipc_case!="table"&&cipc_case!="stack10"&&cipc_case!="hang")
        throw std::runtime_error("Unknown five-scene benchmark case");
    ipc.pcg_data.P_type=1;linear_system_buff_scale=2.0;
    ipc.IPC_dt=.01;ipc.pcg_threshold=1e-3;ipc.Newton_solver_threshold=1e-3;
    ipc.clothYoungModulus=1e6;
    // Remaining material, friction and barrier settings come from the original
    // scene-7 parameterSetting.txt, identically for the two solvers.
    // Benchmark v1: old material for hang/stack; stack geometry/contact remain new.
    if(cipc_case=="hang" || cipc_case=="stack10"){
        ipc.clothThickness=.000318;ipc.clothDensity=472.641509;
        ipc.clothYoungModulus=821000;ipc.bendYoungModulus=821000;
        ipc.PoissonRate=.243;ipc.strainRate=100;
        ipc.IPC_dt=cipc_case=="hang"?.04:.02;
    }
    if(cipc_case=="hang")ipc.frictionRate=ipc.gd_frictionRate=.4;
    gipc::SimpleSceneImporter importer;
    using Transform=Eigen::Transform<double,3,Eigen::Affine>;
    auto add=[&](const std::string& name,int dim,const Transform& t,int role,bool abd=false){
        const int start=tetMesh.vertexNum;
        importer.load_geometry(tetMesh,dim,abd?gipc::BodyType::ABD:gipc::BodyType::FEM,t.matrix(),1e6,
            assets_dir+(dim==3?"tetMesh/":"triMesh/")+name,ipc.pcg_data.P_type,
            abd?BodyBoundaryType::Fixed:BodyBoundaryType::Free);
        const int end=tetMesh.vertexNum;cipc_roles.resize(end,role);
        if(role==0&&!abd)for(int i=start;i<end;++i)tetMesh.boundaryTypies[i]=1;
        cipc_components.push_back({{"label",role==0?"fixed obstacle":"cloth "+std::to_string(role)},
            {"role",role},{"begin",start},{"end",end},{"mesh",name},{"fixed_abd",abd}});
    };
    if(cipc_case=="sphere7"){
        Transform t=Transform::Identity();t.translate(Eigen::Vector3d(0,-.45,0));t.scale(.42);
        add("abd_sphere.msh",3,t,0,true);
    }else if(cipc_case=="bunny"){
        Transform t=Transform::Identity();t.translate(Eigen::Vector3d(0,-.95,0));t.scale(.29167042369955365);
        t.translate(Eigen::Vector3d(.336243149915882,.340252,.030750916089464009));
        add("bunny2.msh",3,t,0,true);
    }else if(cipc_case=="table"){
        // Original C-IPC table surface, uniformly scaled to height .9 m.
        // Center x/z, bottom -.95 m, top -.05 m. Preserve all mesh triangles.
        Transform t=Transform::Identity();t.translate(Eigen::Vector3d(0,-.95,0));
        t.scale(.9/(.974564612-.490013391));
        t.translate(Eigen::Vector3d(-(.222904175-.222851574)/2,-.490013391,-(.354477584-.354517132)/2));
        add("cipc_table.obj",2,t,0);
    }
    if(cipc_case=="stack10"){
        const double rad=3.14159265358979323846/180;
        for(int i=0;i<10;++i){
            Transform t=Transform::Identity();t.translate(Eigen::Vector3d((i%2?1:-1)*.015,-.65+.12*i,0));
            t.rotate(Eigen::AngleAxisd((i-4.5)*3*rad,Eigen::Vector3d::UnitY()));
            t.rotate(Eigen::AngleAxisd((i%2?3:-3)*rad,Eigen::Vector3d::UnitX()));t.scale(.85);
            add("square_cloth_10k.obj",2,t,i+1);
        }
    }else if(cipc_case=="hang"){
        Transform t=Transform::Identity(); // Old 1 m cloth, initial y=0.
        add("cipc_square2K.obj",2,t,1);
        double xmin=1e30,xmax=-1e30,zmin=1e30,zmax=-1e30;
        for(const auto& v:tetMesh.vertexes){xmin=std::min(xmin,v.x);xmax=std::max(xmax,v.x);zmin=std::min(zmin,v.z);zmax=std::max(zmax,v.z);}
        for(int i=0;i<tetMesh.vertexNum;++i){const auto& v=tetMesh.vertexes[i];const double x=(v.x-xmin)/(xmax-xmin),z=(v.z-zmin)/(zmax-zmin);
            if((x<=1e-3&&z<=1e-3)||(x>=1-1e-3&&z>=1-1e-3))tetMesh.boundaryTypies[i]=1;
        }
    }else{
        Transform t=Transform::Identity();t.translate(Eigen::Vector3d(0,.20,0));
        add(cipc_case=="sphere7"?"square_cloth_10k.obj":"square_cloth_23k.obj",2,t,1);
    }
    if(gipc::benchmark::options.cloth_offset_y != 0)
    {
        if(cipc_case != "table" && cipc_case != "bunny")
            throw std::runtime_error("Contact probe supported only for table/bunny");
        for(int i=0; i<tetMesh.vertexNum; ++i)
            if(cipc_roles[i] > 0) tetMesh.vertexes[i].y += gipc::benchmark::options.cloth_offset_y;
    }
    const double probe_vy = std::getenv("CIPC_INITIAL_VY") ? std::stod(std::getenv("CIPC_INITIAL_VY")) : 0;
    if(!std::isfinite(probe_vy)) throw std::runtime_error("Invalid probe velocity");
    if(probe_vy != 0)
    {
        if(cipc_case != "table" && cipc_case != "bunny")
            throw std::runtime_error("Velocity probe supported only for table/bunny");
        for(int i=0; i<tetMesh.vertexNum; ++i)
            if(cipc_roles[i] > 0 && tetMesh.boundaryTypies[i] == 0) tetMesh.velocities[i].y = probe_vy;
    }
    tetMesh.minConer=tetMesh.vertexes.front();tetMesh.maxConer=tetMesh.vertexes.front();
    for(const auto& v:tetMesh.vertexes){
        tetMesh.minConer.x=std::min(tetMesh.minConer.x,v.x);tetMesh.minConer.y=std::min(tetMesh.minConer.y,v.y);tetMesh.minConer.z=std::min(tetMesh.minConer.z,v.z);
        tetMesh.maxConer.x=std::max(tetMesh.maxConer.x,v.x);tetMesh.maxConer.y=std::max(tetMesh.maxConer.y,v.y);tetMesh.maxConer.z=std::max(tetMesh.maxConer.z,v.z);
    }
    // Exactly preserve the established scene-7 bounding box/tolerance scaling.
    if(cipc_case=="sphere7"){tetMesh.minConer=make_double3(-.8,-.87,-.8);tetMesh.maxConer=make_double3(.8,.20,.8);}
    if(cipc_case=="hang"){
        const double dx=tetMesh.maxConer.x-tetMesh.minConer.x;
        const double dy=tetMesh.maxConer.y-tetMesh.minConer.y;
        const double dz=tetMesh.maxConer.z-tetMesh.minConer.z;
        ipc.relative_dhat=.001/std::sqrt(dx*dx+dy*dy+dz*dz);
    }
    xRot=20;yRot=-25;xTrans=0;yTrans=.35;zTrans=-.2;
    if(cipc_case=="hang"){yTrans=.2;zTrans=.3;}
    gipc::Json pinned=gipc::Json::array();for(int i=0;i<tetMesh.vertexNum;++i)if(cipc_roles[i]==0||tetMesh.boundaryTypies[i]!=0)pinned.push_back(i);
    gipc::Json cfg={{"scene",cipc_case},{"components",cipc_components},{"roles",cipc_roles},{"pinned",pinned},
        {"vertices",tetMesh.vertexNum},{"triangles",tetMesh.triangles.size()},{"tets",tetMesh.tetrahedras.size()},
        {"dt",ipc.IPC_dt},{"cloth_volume_density",ipc.clothDensity},{"cloth_areal_density",ipc.clothDensity*ipc.clothThickness},
        {"cloth_young_pa",ipc.clothYoungModulus},{"cloth_thickness",ipc.clothThickness},{"bending_young",ipc.bendYoungModulus},{"nu",ipc.PoissonRate},
        {"strainRate",ipc.strainRate},{"mu",ipc.frictionRate},{"ground_mu",ipc.gd_frictionRate},{"relative_dhat",ipc.relative_dhat},
        {"bounds",{{tetMesh.minConer.x,tetMesh.minConer.y,tetMesh.minConer.z},{tetMesh.maxConer.x,tetMesh.maxConer.y,tetMesh.maxConer.z}}},
        {"camera",{{"xRot",xRot},{"yRot",yRot},{"xTrans",xTrans},{"yTrans",yTrans},{"zTrans",zTrans}}}};
    if(const char* p=std::getenv("CIPC_META_OUT")){std::ofstream f(p);f<<std::setw(2)<<cfg<<'\n';}
    if(const char* p=std::getenv("CIPC_INITIAL_OUT")){std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(tetMesh.vertexes.data()),tetMesh.vertexNum*sizeof(double3));}
    if(const char* p=std::getenv("CIPC_TRIANGLES_OUT")){std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(tetMesh.triangles.data()),tetMesh.triangles.size()*sizeof(uint3));}
    std::cout<<"FIVE_SCENE_READY "<<cipc_case<<" vertices="<<tetMesh.vertexNum<<" pinned="<<pinned.size()<<std::endl;
}

void cipc_record_frame()
{
    if(const char* p=std::getenv("CIPC_ALL_STATES_OUT")){
        CUDA_SAFE_CALL(cudaMemcpy(tetMesh.vertexes.data(),ipc._vertexes,ipc.vertexNum*sizeof(double3),cudaMemcpyDeviceToHost));
        for(const auto& v:tetMesh.vertexes)if(!std::isfinite(v.x)||!std::isfinite(v.y)||!std::isfinite(v.z))throw std::runtime_error("Nonfinite frame state");
        std::ofstream f(p,std::ios::binary|std::ios::app);f.write(reinterpret_cast<const char*>(tetMesh.vertexes.data()),ipc.vertexNum*sizeof(double3));
        if(!f)throw std::runtime_error("Frame output failed");
    }
}
