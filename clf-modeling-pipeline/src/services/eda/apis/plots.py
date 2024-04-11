import pandas as pd
from dataclasses import asdict
from bokeh.models import Panel, Tabs, DataTable, ColumnDataSource, TableColumn, \
    StringFormatter, ScientificFormatter
from bokeh.resources import CDN
from bokeh.embed import file_html
from luntaiDs.ModelingTools.Explore.plots import categ_donut, chart_categ_distr, \
    numeric_donut, chart_histogram, plot_table_profiling
from luntaiDs.ModelingTools.Explore.summary import BinaryStatAttr, NominalCategStatAttr, \
        OrdinalCategStatAttr, NumericStatAttr, BinaryStatObj, NumericStatObj, \
            NominalCategStatObj, OrdinalCategStatObj
from luntaiDs.ModelingTools.FeatureEngineer.preprocessing import TabularPreprocModel
    
    
def table_stat(stats: dict) -> DataTable:
    df = (
        pd.Series(stats)
        .reset_index(name='VALUE')
        .rename(columns = {'index' : 'METRIC'})
    )
    source = ColumnDataSource(df)
    data_table = DataTable(
        source=source, 
        columns=[
            TableColumn(
                field = 'METRIC',
                title = 'METRIC',
                sortable = False,
                formatter = StringFormatter(
                    font_style = 'bold'
                )
            ),
            TableColumn(
                field = 'VALUE',
                title = 'VALUE',
                sortable = False,
                formatter = ScientificFormatter(
                    precision = 5,
                    text_color = 'darkslategrey',
                )
            )
        ], 
        editable=False,
        index_position = None,
        # height = None,
        sizing_mode = 'stretch_both'
    )
    return data_table

def get_nominal_ordinal_plots(stat_obj: NominalCategStatObj | OrdinalCategStatObj):
    donut = categ_donut(
        total = stat_obj.summary.total_,
        missing = stat_obj.summary.missing_,
        size = (500, 400)
    )
    distr = chart_categ_distr(
        vcounts = stat_obj.summary.vcounts_,
        size = (500, 400)
    )
    return {
        'donut' : donut,
        'distr' : distr
    }
    
def get_binary_plots(stat_obj: BinaryStatObj):
    donut = categ_donut(
        total = stat_obj.summary.total_,
        missing = stat_obj.summary.missing_,
        size = (500, 400)
    )
    distr = chart_categ_distr(
        vcounts = stat_obj.summary.binary_vcounts_,
        size = (500, 400)
    )
    return {
        'donut' : donut,
        'distr' : distr
    }
    
def get_numeric_plots(stat_obj: NumericStatObj):
    figs = {}
            
    donut = numeric_donut(
        stat_obj.summary.total_, 
        stat_obj.summary.missing_, 
        stat_obj.summary.zeros_, 
        stat_obj.summary.infs_pos_, 
        stat_obj.summary.infs_neg_, 
        stat_obj.summary.xtreme_,
        size = (600, 500)
    )
    figs['donut'] = donut
    
    tabs_desc_stat = [Panel(
        child = table_stat(asdict(stat_obj.summary.stat_descriptive_)),
        title="Origin"
    )]
    tabs_quant_stat = [Panel(
        child = table_stat(asdict(stat_obj.summary.stat_quantile_)),
        title="Origin"
    )]
    if stat_obj.attr.log_scale_:
        tabs_desc_stat.append(Panel(
            child = table_stat(asdict(stat_obj.summary.stat_descriptive_log_)),
            title="Log"
        ))
        tabs_quant_stat.append(Panel(
            child = table_stat(asdict(stat_obj.summary.stat_quantile_log_)),
            title="Log"
        ))

    tabs_hist = [Panel(
        child = chart_histogram(
            bins_edges = stat_obj.summary.bin_edges_,
            hists = stat_obj.summary.hist_,
            quantile_stat=stat_obj.summary.stat_quantile_,
            desc_stat=stat_obj.summary.stat_descriptive_,
            title = 'Histogram for Valid Values',
            size = (800, 500)
        ),
        title="Histogram - Origin"
    )]
    if stat_obj.attr.log_scale_:
        tabs_hist.append(Panel(
            child = chart_histogram(
                bins_edges = stat_obj.summary.bin_edges_log_,
                hists = stat_obj.summary.hist_log_,
                quantile_stat=stat_obj.summary.stat_quantile_log_,
                desc_stat=stat_obj.summary.stat_descriptive_log_,
                title = 'Histogram for Valid Values',
                size = (800, 500)
            ),
            title="Statistics - Log"
        ))
        
    figs['quant'] = Tabs(tabs=tabs_quant_stat)
    figs['desc'] = Tabs(tabs=tabs_desc_stat)
    figs['hist'] = Tabs(tabs=tabs_hist)
        
    if stat_obj.attr.xtreme_method_ is not None:
        tabs_xtreme = [Panel(
            child = table_stat(asdict(stat_obj.summary.xtreme_stat_)),
            title="Origin"
        )]
        if stat_obj.attr.log_scale_:
            tabs_xtreme.append(Panel(
                child = table_stat(asdict(stat_obj.summary.xtreme_stat_log_)),
                title="Log"
            ))
        
        figs['xtreme'] = Tabs(tabs=tabs_xtreme)
        
    return figs

def plot_table_profiling_to_html_string(tpm: TabularPreprocModel) -> str:
    ts = tpm.to_tabular_stat()
    plot = plot_table_profiling(ts)
    return file_html(plot, CDN, "Feature Standalone Stat Summary")