import argparse

class StoreInDict(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        d = getattr(namespace, self.dest)
        for opt in values:
            k,v = opt.split("=", 1)
            k = k.lstrip("-")
            d[k] = v
        setattr(namespace, self.dest, d)

# Prevent argparse from trying to distinguish between positional arguments
# and optional arguments. Yes, it's a hack.
p = argparse.ArgumentParser( prefix_chars=' ' )

# Put all arguments in a single list, and process them with the custom action above,
# which convertes each "--key=value" argument to a "(key,value)" tuple and then
# merges it into the given dictionary.
p.add_argument("options", nargs="*", action=StoreInDict, default=dict())

args = p.parse_args()
print(args.options)
