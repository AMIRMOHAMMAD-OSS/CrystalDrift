"""Freeze one-pass objective controls and the matched initial network."""
from register import model,freeze,V3,digest
a=[model("meanflow-small-11","small-meanflow-laplace-11"),model("meanflow-medium-11","medium-meanflow-laplace-11"),
   model("contacts-medium-11","medium-contacts-laplace-11")]
initial=V3/"models/controls/untrained-small-11/initial.pt"
a.append(dict(tag="untrained-small",checkpoint=str(initial.relative_to(V3)).replace(chr(92),"/"),sha=digest(initial)))
freeze("controlled-frozen",a)

