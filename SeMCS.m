classdef SeMCS < semcs.LegacyEngine
    methods
        function obj=SeMCS(sampler),setup_semcs();obj@semcs.LegacyEngine(sampler);end
        function out=RunIte(obj)
            if isa(obj.IntBay,'SeMIS_Bay_Nataf')&&SeMISPaperARSPolicy.enabled(obj.IntBay)
                validateattributes(obj.IntBay.G.p,{'numeric'},{'scalar','finite','>',0,'<',1});
                validateattributes(obj.IntBay.G.run.NumSam,{'numeric'},{'scalar','finite','integer','positive'});
                assert(isempty(obj.IntBay.G.run.schedule),'SeMISPaperARS:Schedule','This adaptive policy does not accept a prescribed G schedule.');
                assert(obj.IntBay.G.run.MaxIte==obj.IntBay.G.paper_controls.legacy_MaxIte, ...
                    'SeMISPaperARS:Cap','Change the registered cap controls consistently; MaxIte alone does not define this policy.');
                % Preserve the model-owned cap policy; LegacyEngine otherwise
                % forces its historical select_at_cap=true compatibility rule.
                engine=semcs.Engine(obj.IntBay,obj.Sampler);out=engine.run();
                out=SeMISPaperARSPolicy.annotate(out);
            else
                out=RunIte@semcs.LegacyEngine(obj);
            end
        end
    end
end
