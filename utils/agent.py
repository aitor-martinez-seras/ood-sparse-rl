import torch

import utils
from .other import device
from model import *

# from PIL import Image
import numpy as np
from skimage.transform import resize


def select_explainer(explainer_name: str, model: nn.Module):
    from captum.attr import IntegratedGradients, LayerGradCam, Saliency

    # TODO: Si se requieren muchas transformaciones sobre la atribucion, 
    #   se puede tener un atributo dentro del agente que contenga dicha transformacion
    #   y que se cree en esta funcion devolviendo tanto el explainer como la funcion de transformacion
    if explainer_name == "saliency":
        explainer = Saliency(model)

    elif explainer_name == "gradcam":
        explainer = LayerGradCam(model, model.image_conv[0])
        # transformation = transform_ftmap_sized_heatmap_to_orig_size()
    
    elif explainer_name == "integrated_gradients":
        explainer = IntegratedGradients(model)
    
    else:
        raise ValueError(f"Unknown explainer {explainer_name}")

    return explainer


class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, obs_space, action_space, model_dir,
                 argmax=False, num_envs=1, use_memory=False, use_text=False,
                 separated_networks=False, explainer=None):
        
        obs_space, self.preprocess_obss = utils.get_obss_preprocessor(obs_space,evaluation=True)
        self.separated_networks = separated_networks
        self.use_recurrence = use_memory
        self.action_space = action_space
        self.argmax = argmax
        self.num_envs = num_envs
        self.explainer = explainer  # By default is none

        print('Model directory:',model_dir)
        print('self.separated_networks',self.separated_networks)
        print('Action space:',action_space)

        if self.separated_networks:
            actor = ActorModel_RAPID(obs_space, action_space, use_memory=use_memory)
            critic = CriticModel_RAPID(obs_space, action_space, use_memory=use_memory)

            actor.load_state_dict(utils.get_model_state(model_dir=model_dir,separated_network='actor'))
            critic.load_state_dict(utils.get_model_state(model_dir=model_dir,separated_network='critic'))

            actor.to(device)
            critic.to(device)

            actor.eval()
            critic.eval()

            self.acmodel = (actor, critic)
        else:
            self.acmodel = ACModelRIDE(obs_space, action_space, use_memory=use_memory, use_text=use_text)
            self.acmodel.load_state_dict(utils.get_model_state(model_dir))
            self.acmodel.to(device)
            self.acmodel.eval()
            if explainer:
                self.explainer = select_explainer(explainer, self.acmodel)

        print('Model LOADED!')

        if self.use_recurrence:
            self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size, device=device)


        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir))

    def get_actions(self, obss, ood_method='', gen_explanation=None):
        preprocessed_obss = self.preprocess_obss(obss, device=device)

        if self.explainer:
            if self.use_recurrence:
                raise NotImplementedError
            else:
                if self.separated_networks:
                    raise NotImplementedError
                else:
                    # TODO: Voy a hacer que el modelo reciba directamente la imagen como tensor y no el objeto DictList
                    if ood_method:
                        dist, activations, vext = self.acmodel(preprocessed_obss.image, ood_method=ood_method)
                    else:
                        dist, vext = self.acmodel(preprocessed_obss.image)
                    
                    # TODO: Plotear usando el highlighting y asi lo veo overlapeado
                    # TODO: Tengo que decidir el rango en el que ploteo (normalizar heatmap consigo mismo,
                    #   normalizar todos los heatmaps cogiendo maximos y minimos de muchos heatmaps, etc)
                    # TODO: Tambien tengo que probar a plotear otras capas de la parte convolucional
                    
                    # In this case we are always sampling
                    actions = dist.sample()

                    # Retrieve the attribution
                    self.acmodel.explanation = True
                    # With GradCam if Relu attributions is True, then only positive attributions are returned. Otherwise
                    # both positive and negative attributions are returned. In that case we could use the absolute value.
                    attribution = self.explainer.attribute(preprocessed_obss.image, target=actions, relu_attributions=True)
                    self.acmodel.explanation = False

                    # Reformat attribution
                    attribution = resize(attribution[0,0].detach().numpy(), (7,7))
                    # Scale the attribution into the [0, 0.5] range
                    attribution = 0.5 * (attribution - attribution.min()) / (attribution.max() + attribution.min())

                    # Gradcam to size
                    # from PIL import Image
                    # import numpy as np
                    # import matplotlib.pyplot as plt
                    # pil_attr = Image.fromarray(attribution[0,0].detach().numpy())
                    # pil_attr_resized = pil_attr.resize((7,7))
                    # attr_resized = np.array(pil_attr_resized)
                    # fix, ax = plt.subplots(1,1)
                    # im = ax.imshow(attr_resized[...,None]*100000, alpha=0.5)
                    # plt.colorbar(im)
                    # plt.show()

                    # import matplotlib.pyplot as plt
                    
                    # ax.imshow(preprocessed_obss.image[0])
                    # plt.show()

                    # att_norm = torch.sum(attribution[0], axis=2)
                    # fix, ax = plt.subplots(1,1)
                    # im = ax.imshow(att_norm*10000, alpha=0.5)
                    # plt.colorbar(im)
                    # plt.show()


        else:
            with torch.no_grad():
                if self.use_recurrence:
                    if self.separated_networks:
                        dist,mem = self.acmodel[0](preprocessed_obss)
                        vext,mem = self.acmodel[1](preprocessed_obss)
                    else:
                        dist, vext, self.memories = self.acmodel(preprocessed_obss,self.memories)
                else:
                    if self.separated_networks:
                        if ood_method:
                            raise NotImplementedError
                        dist = self.acmodel[0](preprocessed_obss)
                        vext = self.acmodel[1](preprocessed_obss)
                    else:
                        if ood_method:
                            dist, activations, vext = self.acmodel(preprocessed_obss, ood_method=ood_method)
                        else:
                            dist, vext = self.acmodel(preprocessed_obss)

            # In case of using the explainer, the action is previously sampled
            if self.argmax:
                actions = dist.probs.max(1, keepdim=True)[1]
            else:
                actions = dist.sample()

        if ood_method:
            return actions.cpu().numpy(), activations, vext
        # TODO: Hay que mejorar la forma de gestionar explicabilidad
        elif self.explainer:
            return actions.cpu().numpy(), attribution, vext, attribution

        else:
            return actions.cpu().numpy(), vext

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        if self.use_recurrence:
            masks = 1 - torch.tensor(dones, dtype=torch.float, device=device).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])
